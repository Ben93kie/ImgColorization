import argparse
import torch
from PIL import Image
import cv2
import torchvision.transforms as T


from utils.checkpoint import ColorizationCheckpointer
from utils.qualitative import save_predictions
from cfg import _C as cfg
from models.build_model import build_model


class Evaluator:
    def __init__(self, cfg, model_to_load=''):
        self.model = build_model(cfg)
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.arguments = {}
        self.arguments["iteration"] = 0
        self.color_space = cfg.INPUT.COLOR_SPACE
        output_dir = cfg.OUTPUT_DIR
        self.input_transform = T.Compose([T.Resize(size=(256,256)),
                                          T.Grayscale(),
                                          T.ToTensor(),
                                          T.Normalize(cfg.INPUT.PIXEL_MEAN_G, cfg.INPUT.PIXEL_STD),
                                          ])
        # Use this on target images(colorful ones)
        self.target_transform = T.Compose([T.Resize(size=(256,256)),
                                           T.ToTensor(),
                                           T.Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD),
                                           ])
        self.target_transform_g = T.Compose([T.Resize(size=(256,256)),
                                             T.ToTensor(),
                                             T.Normalize(cfg.INPUT.PIXEL_MEAN_G, cfg.INPUT.PIXEL_STD),
                                             ])
        self.checkpointer = ColorizationCheckpointer(
            cfg, self.model, None, None, output_dir, False, model_to_load=model_to_load)
        self.extra_checkpoint_data = self.checkpointer.load()

    def inference(self, img_path):
        self.model.eval()
        with torch.no_grad():

            if self.color_space == 'RGB':
                img = Image.open(img_path)
                gray = self.input_transform(img)
                gray = torch.unsqueeze(gray, 0).cuda()
            elif self.color_space == 'LAB':
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                img = cv2.resize(img, (256, 256))
                img = torch.tensor(img)
                img = img / 255.
                gray = img[:, :, 0]
                gray = (gray - 0.5) / 0.5
                gray = torch.unsqueeze(gray, 0)
                gray = torch.unsqueeze(gray, 0).cuda()
            else:
                raise ValueError("Color space not yet implemented: ", self.color_space)

            predictions_val = self.model(gray)

            if cfg.TEST.SAVE_SAMPLE_IMGS:
                if cfg.INPUT.COLOR_SPACE == 'RGB':
                    save_predictions([gray,predictions_val],'demo',cfg)
                if cfg.INPUT.COLOR_SPACE == 'LAB':
                    predictions_val3channel = torch.cat((gray,predictions_val),dim=1)
                    save_predictions([gray,predictions_val3channel],'demo',cfg)
                print("output image saved as demo.jpg")
        if cfg.INPUT.COLOR_SPACE == 'RGB':
            return predictions_val
        elif cfg.INPUT.COLOR_SPACE == 'LAB':
            return predictions_val3channel
        else:
            raise ValueError("Only RGB and LAB color space available, not: ", cfg.INPUT.COLOR_SPACE)


def main():
    parser = argparse.ArgumentParser(description="PyTorch Image Colorization - Inference Script")
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
        "--img_path",
        default="",
        help="img path",
    )
    args = parser.parse_args()
    torch.manual_seed(0)
    if args.config_file == '':
        print("No config file given. Trying with default config 'configs/larger_net_pretrained.yaml'")
        args.config_file = "configs/larger_net_pretrained.yaml"
    cfg.merge_from_file(args.config_file)
    cfg.TEST.SAMPLE_IMGS_PATH = '.'
    cfg.freeze()
    if args.img_path == '':
        print("You did not provide an image path. Running this script with example image 'test_img.jpg'")
        args.img_path = "test_img.jpg"

    model = Evaluator(cfg, model_to_load=args.model_to_load)
    pred_img = model.inference(args.img_path)




if __name__ == "__main__":
    main()