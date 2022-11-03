import argparse
import os.path
from pathlib import Path

import cv2
import torch
import torchvision.transforms.functional as F

from core.parse import update_cfg
from core.post_process import PostProcess
from utils.tools import load_yaml, cv2_read_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default='', help="yaml path")
    parser.add_argument("--ckpt", type=str, default='', help="model checkpoint path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_yaml(filepath=[args.cfg])
    cfg = update_cfg(cfg, device, use_dataset=False)

    model = cfg["model"]
    model.to(device=device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    test_images = cfg["Test"]["test_pictures"]
    for img in test_images:
        image_path = img
        image_name = Path(image_path).stem
        print(f"Reading image: {image_path}")

        img = F.to_tensor(cv2_read_image(img))
        # # resize to fixed input size
        # input_size = cfg["Train"]["input_size"][1:]
        # original_size = [img.size(1), img.size(2)]
        # img = F.resize(img, size=input_size)

        with torch.no_grad():
            result = PostProcess(img, model, device).fcn()

        result = torch.squeeze(result, dim=0)
        # result = torch.permute(result, dims=[2, 0, 1])   # (h, w, c) -> (c, h, w)
        # # back to original size
        # result = F.resize(result, size=original_size)
        # result = torch.permute(result, dims=[1, 2, 0])   # (c, h, w) -> (h, w, c)
        result = result.cpu().numpy()
        # rgb -> bgr
        result = result[..., ::-1]
        print(f"Writing result of {image_path}")
        cv2.imwrite(os.path.join(cfg["Test"]["test_results"], f"segmentation-{image_name}.jpg"), result)


if __name__ == '__main__':
    main()
