# %%
from torch import autocast
from diffusers import StableDiffusionPipeline, AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

import torch
from tqdm import tqdm
import os
import socket
import argparse
import logging
import time
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
logging.getLogger("tqdm").setLevel(logging.WARNING)


model_id = "runwayml/stable-diffusion-v1-5"

pipeline_T2I = StableDiffusionPipeline.from_pretrained(
    model_id,
    use_auth_token=True
).to("cuda")


pipeline_I2I = AutoPipelineForImage2Image.from_pretrained(
    model_id,
    use_auth_token=True
).to("cuda")


# if one wants to set `leave=False`
pipeline_T2I.set_progress_bar_config(leave=False)
pipeline_I2I.set_progress_bar_config(leave=False)
# if one wants to disable `tqdm`
pipeline_T2I.set_progress_bar_config(disable=True)
pipeline_I2I.set_progress_bar_config(disable=True)

###########
IMAGE_SIZE = 512

if socket.gethostname() == "PC":
    IMAGE_PATH = "/home/xxx/work/datasets/COCO_2017/val2017_resize/512"
    SAVE_PATH = F"/home/xxx/work/datasets/COCO_2017/AIGC_EVAL/full/SD_V1.5"
    cache_dir = "/home/xxx/.cache/huggingface/hub"


else:
    IMAGE_PATH = "/home/special/user_new/xxx/datasets/COCO_2017/val2017_resize/512"
    SAVE_PATH = F"/home/special/user_new/xxx/datasets/COCO_2017/AIGC_EVAL/AIGI/full/SD_V1.5"
    cache_dir = "/home/special/user_new/xxx/pretrained/huggingface"


data = torch.load("../data/captions_dict.pp")


def generate(args):

    num = 0
    total = len(data.keys())
    for key in tqdm(data.keys(), disable=True):
        num += 1
        start_time = time.time()

        caption_id = key
        caption = data[key]["caption"]
        image_id = data[key]["image_id"]
        image_name = data[key]["image_name"]

        while len(image_name) != 16:
            image_name = "0"+image_name

        image_path = F"{IMAGE_PATH}/{image_name}"


        for type in ["T2I", "I2I", "TI2I"]:
            args.type = type

            # ###
            exist_filename = F"{SAVE_PATH}/{IMAGE_SIZE}/{args.type}/{image_id}_{caption_id}.jpg"

            if os.path.exists(exist_filename):
                continue
            # ###

            new_filename = F"{SAVE_PATH}/{IMAGE_SIZE}/{args.type}/{image_id}_{caption_id}.jpg"
            print(new_filename)

            if args.type == "T2I":
                ouput = pipeline_T2I(prompt=caption)
                image = ouput["images"][0]
                image.save(new_filename)
            elif args.type == "I2I":
                init_image = load_image(image_path)
                ouput = pipeline_I2I(prompt="", image=init_image)
                image = ouput["images"][0]
                image.save(new_filename)
            elif args.type == "TI2I":
                init_image = load_image(image_path)
                ouput = pipeline_I2I(prompt=caption, image=init_image)
                image = ouput["images"][0]
                image.save(new_filename)
            else:
                raise ValueError(f"{args.type} is not a valid item")

        end_time = time.time()
        print(F"{num}/{total}\t{end_time-start_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--type', type=str, default='T2I',
                        help='')  # ["T2I","I2I","TI2I"]

    args = parser.parse_args()

    generate(args)
