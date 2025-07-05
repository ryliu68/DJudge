
from diffusers import DiffusionPipeline, AutoPipelineForImage2Image
import torch
from diffusers.utils import make_image_grid, load_image
import torch
from tqdm import tqdm
import os
import socket
import argparse
import logging
import time

logging.getLogger("tqdm").setLevel(logging.WARNING)


seed = 2024
torch.manual_seed(seed)

###########
IMAGE_SIZE = 1024
model_id = "stabilityai/stable-diffusion-xl-base-1.0"


if socket.gethostname() == "PC":
    IMAGE_PATH = F"/home/xxx/work/datasets/COCO_2017/val2017_resize/{IMAGE_SIZE}"
    SAVE_PATH = F"/home/xxx/work/datasets/AIGC_EVAL/AIGI/full/SD_XL"
    cache_dir = "/home/xxx/.cache/huggingface/hub"

else:
    IMAGE_PATH = F"/home/special/user_new/xxx/datasets/COCO_2017/val2017_resize/{IMAGE_SIZE}"
    SAVE_PATH = F"/home/special/user_new/xxx/datasets/COCO_2017/AIGC_EVAL/AIGI/full/SD_XL"
    cache_dir = "/home/special/user_new/xxx/pretrained/huggingface/hub"


###########
data = torch.load("../data/captions_dict.pp")


def generate(args):
    #######################################################
    if args.type == "T2I":
        pipeline_T2I = DiffusionPipeline.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        ).to("cuda")

        ###
        # if one wants to set `leave=False`
        pipeline_T2I.set_progress_bar_config(leave=False)
        # if one wants to disable `tqdm`
        pipeline_T2I.set_progress_bar_config(disable=True)

    else:
        pipeline_I2I = AutoPipelineForImage2Image.from_pretrained(
            model_id,
            cache_dir=cache_dir
        ).to("cuda")

        ###
        # if one wants to set `leave=False`
        pipeline_I2I.set_progress_bar_config(leave=False)
        # if one wants to disable `tqdm`
        pipeline_I2I.set_progress_bar_config(disable=True)

    ########################################################

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
        # print(image_path)

        new_filename = F"{SAVE_PATH}/{IMAGE_SIZE}/{args.type}/{image_id}_{caption_id}.jpg"

        if os.path.exists(new_filename):
            # print(new_filename)
            continue

        if args.type == "T2I":
            # ouput = pipeline_T2I(prompt=caption,height=512,width=512)
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
