import cv2
import torch
import numpy as np
import os

def save_predictions(imgs,iteration,cfg):
    normalize_mean = torch.tensor(cfg.INPUT.PIXEL_MEAN,device='cuda')
    normalize_std = torch.tensor(cfg.INPUT.PIXEL_STD,device='cuda')
    normalize_mean_g = torch.tensor(cfg.INPUT.PIXEL_MEAN_G,device='cuda')
    normalize_std_g = torch.tensor(cfg.INPUT.PIXEL_STD_G,device='cuda')
    color_space  = cfg.INPUT.COLOR_SPACE
    cv2img=[]
    for img in imgs:
        if color_space == 'RGB':
            if img.shape[1]==1:
                ndarr = img.mul(torch.unsqueeze(torch.unsqueeze(normalize_mean_g, 1), 1)). \
                        add(torch.unsqueeze(torch.unsqueeze(normalize_std_g, 1), 1)).mul(255). \
                        clamp_(0, 255)[0, :, :, :].permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                ndarr = cv2.cvtColor(ndarr, cv2.COLOR_GRAY2BGR)
                ndarr = cv2.cvtColor(ndarr, cv2.COLOR_BGR2RGB)
            elif img.shape[1]==3:
                ndarr = img.mul(torch.unsqueeze(torch.unsqueeze(normalize_mean, 1), 1)). \
                        add(torch.unsqueeze(torch.unsqueeze(normalize_std, 1), 1)).mul(255). \
                        clamp_(0, 255)[0, :, :, :].permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                cv2.cvtColor(ndarr, cv2.COLOR_BGR2RGB)
                ndarr = cv2.cvtColor(ndarr, cv2.COLOR_BGR2RGB)
        elif color_space == 'LAB':
            if img.shape[1] == 1:
                ndarr = img.mul(torch.unsqueeze(torch.unsqueeze(normalize_mean_g, 1), 1)). \
                        add(torch.unsqueeze(torch.unsqueeze(normalize_std_g, 1), 1)).mul(255). \
                        clamp_(0, 255)[0, :, :, :].permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                ndarr = cv2.cvtColor(ndarr, cv2.COLOR_GRAY2BGR)
                ndarr = cv2.cvtColor(ndarr, cv2.COLOR_BGR2RGB)
            elif img.shape[1] == 3:
                ndarr = img.mul(torch.unsqueeze(torch.unsqueeze(normalize_mean, 1), 1)). \
                        add(torch.unsqueeze(torch.unsqueeze(normalize_std, 1), 1)).mul(255). \
                        clamp_(0, 255)[0, :, :, :].permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                ndarr = cv2.cvtColor(ndarr, cv2.COLOR_LAB2BGR)

        ndarr = np.ascontiguousarray(ndarr, dtype=np.uint8)

        cv2img.append(ndarr)
    all_imgs = np.hstack(cv2img)
    if not os.path.exists(cfg.TEST.SAMPLE_IMGS_PATH):
        os.mkdir(cfg.TEST.SAMPLE_IMGS_PATH)
    write_path = os.path.join(cfg.TEST.SAMPLE_IMGS_PATH, str(iteration)+'.jpg')
    cv2.imwrite(write_path,all_imgs)



