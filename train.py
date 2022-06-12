import datetime
import torch
import os
import argparse
import logging
import time
import torch.distributed as dist
from tqdm import tqdm
import torch.nn as nn

from utils.comm import get_world_size, synchronize, get_rank
from utils.miscellaneous import mkdir, save_config, cfg_node_to_dict
from utils.logger import setup_logger
from utils.metric_logger import MetricLogger
from utils.checkpoint import ColorizationCheckpointer
from utils.qualitative import save_predictions
from cfg import _C as cfg
from models.build_model import build_model
from optimizer.build import make_optimizer, make_lr_scheduler
from data.build import make_data_loader



def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses



class Trainer:
    def __init__(self, cfg, local_rank, distributed, model_to_load='', data_dir=''):
        raw_cfg = cfg_node_to_dict(cfg)
        self.model = build_model(cfg)
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.optimizer = make_optimizer(cfg, self.model)
        self.scheduler = make_lr_scheduler(cfg, self.optimizer)
        if distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank], output_device=local_rank,
                broadcast_buffers=False,
            )
        self.arguments = {}
        self.arguments["iteration"] = 0

        output_dir = cfg.OUTPUT_DIR

        save_to_disk = get_rank() == 0

        self.checkpointer = ColorizationCheckpointer(
            cfg, self.model, self.optimizer, self.scheduler, output_dir, save_to_disk, model_to_load=model_to_load)
        self.extra_checkpoint_data = self.checkpointer.load()
        self.arguments.update(self.extra_checkpoint_data)
        self.data_loader = make_data_loader(
            cfg,
            is_train=True,
            is_distributed=distributed,
            start_iter=self.arguments["iteration"],
            data_dir=data_dir
        )
        self.test_period = cfg.SOLVER.TEST_PERIOD

        if self.test_period != 0:
            self.data_loader_val = make_data_loader(cfg, is_train=False, is_distributed=distributed, is_for_period=True)
        else:
            self.data_loader_val = None

        self.checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
        if cfg.SOLVER.LOSS == 'L1':
            self.loss = nn.L1Loss()
        elif cfg.SOLVER.LOSS == 'L2':
            self.loss = nn.MSELoss()
        else:
            raise ValueError("Supporting only L1 and L2 loss, not: ", self.loss)

    def train(self):

        logger = logging.getLogger("ImgColorization.train")
        logger.info("Start training")
        meters = MetricLogger(delimiter="  ")
        max_iter = len(self.data_loader)
        print("number of images", len(self.data_loader.sampler.data_source.ids))
        num_images = len(self.data_loader.sampler.data_source.ids)
        number_epochs_to_train = cfg.SOLVER.MAX_ITER_EPOCH
        if number_epochs_to_train > 0:
            max_iter = num_images * number_epochs_to_train // cfg.SOLVER.IMS_PER_BATCH
            print("train for ", max_iter, " iterations, i.e. ", number_epochs_to_train, " epochs")
        save_after_epochs = True if cfg.SOLVER.CHECKPOINT_PERIOD_EPOCH > 0 else False
        start_iter = 0
        self.model.train()
        start_training_time = time.time()
        end = time.time()

        for iteration, (images, targets ) in enumerate(self.data_loader, start_iter):

            data_time = time.time() - end
            iteration = iteration + 1
            self.arguments["iteration"] = iteration

            images = images.to(self.device)
            targets = targets.to(self.device)

            predictions = self.model(images)

            loss = self.loss(predictions,targets)
            loss_dict = {'ownloss':loss}
            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            self.scheduler.step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            if iteration % 20 == 0 or iteration == max_iter:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=self.optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )

            if iteration % self.checkpoint_period == 0 or (
                    save_after_epochs and
                    iteration % (num_images * cfg.SOLVER.CHECKPOINT_PERIOD_EPOCH // cfg.SOLVER.IMS_PER_BATCH) == 0):
                save_path_append = 'models/' + cfg.DATASETS.TRAIN[0]
                if not os.path.exists(save_path_append):
                    os.makedirs(save_path_append)
                    print("Created " + save_path_append)
                self.checkpointer.save(save_path_append + '/' + cfg.DATASETS.TRAIN[0] + "model_{:07d}".format(iteration),
                                  **self.arguments)
            if self.test_period < 0:
                self.test_period = num_images // cfg.SOLVER.IMS_PER_BATCH * (-self.test_period)
            if self.data_loader_val is not None and self.test_period > 0 and iteration % self.test_period == 0:
                self.validate()
            if iteration == max_iter:
                break

    def validate(self):
        meters_val = MetricLogger(delimiter="  ")
        synchronize()
        self.model.eval()
        with torch.no_grad():
            # Should be one image for each GPU:
            for iteration_val, (images_val, targets_val) in enumerate(tqdm(self.data_loader_val)):
                images_val = images_val.to(self.device)
                targets_val = targets_val.to(self.device)
                predictions_val = self.model(images_val)

                if cfg.TEST.SAVE_SAMPLE_IMGS:
                    if cfg.INPUT.COLOR_SPACE == 'RGB':
                        save_predictions([images_val,targets_val,predictions_val],iteration_val,cfg)
                    if cfg.INPUT.COLOR_SPACE == 'LAB':
                        targets_val3channel = torch.cat((images_val,targets_val),dim=1)
                        predictions_val3channel = torch.cat((images_val,predictions_val),dim=1)
                        save_predictions([images_val,targets_val3channel,predictions_val3channel],iteration_val,cfg)

                loss = self.loss(predictions_val, targets_val)
                loss_dict = {'ownloss': loss}

                loss_dict_reduced = reduce_loss_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                meters_val.update(loss=losses_reduced, **loss_dict_reduced)
        synchronize()
        logger = logging.getLogger("ImgColorization.val")
        logger.info("Start validating")
        logger.info(
            meters_val.delimiter.join(
                [
                    "[Validation]: ",
                    "{meters}",
                    "lr: {lr:.6f}",
                    "max mem: {memory:.0f}",
                ]
            ).format(
                meters=str(meters_val),
                lr=self.optimizer.param_groups[0]["lr"],
                memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
            )
        )


def main():
    parser = argparse.ArgumentParser(description="PyTorch Image Colorization")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--model_to_load",
        default="",
        help="model to be loaded for evaluation",
    )
    parser.add_argument(
        "--data_dir",
        default="",
        help="data directory",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    torch.manual_seed(0)
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("ImgColorization", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)


    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    # logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    model = Trainer(cfg, args.local_rank, args.distributed, model_to_load=args.model_to_load, data_dir=args.data_dir)
    model.train()




if __name__ == "__main__":
    main()
