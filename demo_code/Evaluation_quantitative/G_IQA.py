from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image


import argparse

import numpy as np
import torch
from tqdm import tqdm

import json

import argparse
import torch
import warnings
from tqdm import tqdm


import torch


from util import multidict


from eval_utils import model_configs,  fix_seed


warnings.filterwarnings("ignore")


device = 'cuda' if torch.cuda.is_available() else 'cpu'

BATH_PATH = "/home/xxx/work/datasets/COCO_2017/AIGC_EVAL/FID_4971"


def test(args, processor, model):

    filenames = []
    pred_labels = []
    for i in tqdm(range(len(args.filenames)), desc=F"{args.AIGC}-{args.sub_set}", disable=True):
        filename = args.filenames[i]
        caption = args.captions[i]

        image = Image.open(F"{args.prefix_path}/{filename}")

        if args.task == "object":
            prompt = "Question: what the object in there? only return the class name in COCO dataset, i.e. the noun. Answer:"
        elif args.task == "align":
            prompt = F"Question: the image content align very well with {caption}? only return 'yes' or 'no'. Answer:"
        elif args.task == "prob":
            prompt = F"Question: what's the probability of the image content well align with {caption}? only return a float number. Answer:"
        else:
            raise ValueError

        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

        generated_ids = model.generate(**inputs)

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        filenames.append(filename)
        pred_labels.append(generated_text)

    return filenames, pred_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Glow on CIFAR-10')

    def str2bool(s):
        return s.lower().startswith('t')

    parser.add_argument('--batch_size', default=64,
                        type=int, help='Batch size per GPU')
    parser.add_argument('--benchmark', type=str2bool,
                        default=True, help='Turn on CUDNN benchmarking')

    parser.add_argument('-I', "--indicator", type=str, default="blip2")

    parser.add_argument('-T', '--task', required=True, default="object", choices=["object", "align", "prob"], type=str, help='')

    args = parser.parse_args()

    fix_seed(2024)

    args.image_filenames_caption_json = F"data/image_filename-caption.json"
    args.image_num = '4971'

    with open(args.image_filenames_caption_json, 'r') as file:
        args.img_data = json.load(file)

    args.filenames = args.img_data["filenames"]
    args.captions = args.img_data["captions"]

    cache_dir = "/home/xxx/work/Weights/blip2"

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=cache_dir)  # BLIP-2
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", cache_dir=cache_dir)

    model.to(device)

    AIGCs = ['ORG', "SD_V14", "SD_V15", "Versatile",
             "SD_V21", "SD_XL", "GALIP", "DFGAN", "Dalle2", "Dalle3"]

    args.nr_results = multidict()

    nr_results_path = F"results/nr_4971/{args.indicator}_{args.task}.json"

    for AIGC in AIGCs:
        args.AIGC = AIGC

        if AIGC == "ORG":
            args.sub_set = ""
            # ORG
            args.IMAGE_SIZE = 512
            args.prefix_path = F"{BATH_PATH}/ORG/{args.IMAGE_SIZE}"

            _filenames, scores = test(args, processor, model)

            args.nr_results[args.image_num][args.indicator]["ORG"] = {
                "name_scores": dict(zip(_filenames, scores))}

            with open(F"results/nr_4971/{args.indicator}_ORG.json", 'w') as json_file:
                json.dump(args.nr_results, json_file)
            # ORG
        else:
            args.IMAGE_SIZE = model_configs[AIGC]["image_size"]
            args.sub_sets = model_configs[AIGC]["sub_sets"]

            for sub_set in tqdm(args.sub_sets, desc=F"{AIGC}"):
                args.sub_set = sub_set

                args.prefix_path = F"{BATH_PATH}/{AIGC}/{args.IMAGE_SIZE}/{sub_set}"
                _filenames, scores = test(args, processor, model)

                args.nr_results[args.image_num][args.indicator][AIGC][sub_set] = {
                    "name_scores": dict(zip(_filenames, scores))}

            with open(nr_results_path, 'w') as json_file:
                json.dump(args.nr_results, json_file)
