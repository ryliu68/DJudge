#!/usr/bin/env python

from openai import OpenAI
import requests
from io import BytesIO
from PIL import Image
import argparse
import json
from tqdm import tqdm
import time

from PIL import Image
from io import BytesIO
import os

openai = OpenAI(
    api_key="xxx")

model = "dall-e-3"


caption_path = "../../data/image_captions_dict_new.json"

IMAGE_PATH = "/home/xxx/work/datasets/COCO_2017/val2017_resize/1024"


SAVE_PATH = "/home/xxx/work/datasets/COCO_2017/AIGC_EVAL/full/Dalle3"
IMAGE_SIZE = 1024


def to_png(image_path):

    with open(image_path, 'rb') as f:
        jpg_data = f.read()

    jpg_img = Image.open(BytesIO(jpg_data)).convert("RGB")

    png_output = BytesIO()

    jpg_img.save(png_output, format='PNG')

    png_output.seek(0)
    png_data = png_output.read()

    return png_data


def t2i(prompt):
    # Generate an image based on the prompt
    response = openai.images.generate(
        prompt=prompt, model=model, size="1024x1024")

    return response


def i2i(image_path):

    image = to_png(image_path)

    response = openai.images.create_variation(
        image=image, model=model, n=1, size='1024x1024')

    return response


def ti2i(prompt, image_path):

    image = to_png(image_path)
    mask = open("mask.png", "rb")

    response = openai.images.edit(
        image=image, prompt=prompt, mask=mask, model=model, n=1, size='1024x1024')

    return response


def generate(args):
    with open(caption_path) as file:
        data = json.load(file)

    keys = data.keys()

    num = 0
    total = len(keys)
    for key in tqdm(keys, disable=True):
        num += 1
        start_time = time.time()

        image_id = data[key]['image_id']
        image_name = data[key]['image_name']

        captions = data[key]['captions']

        caption_id = captions[0]['caption_id']
        caption = captions[0]['caption']

        while len(image_name) != 16:
            image_name = "0"+image_name

        image_path = F"{IMAGE_PATH}/{image_name}"

        # ###
        new_filename = F"{SAVE_PATH}/{IMAGE_SIZE}/{args.type}/{image_id}_{caption_id}.jpg"
        # print(new_filename)

        if os.path.exists(new_filename):
            continue
        # ###
        try:
            if args.type == "T2I":
                response = t2i(prompt=caption)
            elif args.type == "I2I":
                response = i2i(image_path=image_path)
            elif args.type == "TI2I":
                response = ti2i(prompt=caption, image_path=image_path)
            else:

                raise ValueError(f"{args.type} is not a valid item")

            # extract image URL from response
            generated_image_url = response.data[0].url
            generated_image = requests.get(
                generated_image_url).content  # download the image

            with open(new_filename, "wb") as image_file:
                # write the image to the file
                image_file.write(generated_image)

        except Exception as e:
            print(
                f"An unexpected error occurred: {e} in generating: {num}/{total}\t{new_filename}")
            continue

        end_time = time.time()
        print(F"{num}/{total}\t{end_time-start_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--type', type=str, required=True, choices=["T2I"],
                        help='')  #

    args = parser.parse_args()

    print(args)

    generate(args)
