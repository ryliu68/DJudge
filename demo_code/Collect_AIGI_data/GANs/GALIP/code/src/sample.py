import torch
import os
from PIL import Image
import clip
import os.path as osp
import os, sys
import torchvision.utils as vutils


import os
import PIL

import numpy as np


from tqdm import tqdm
import glob
import socket
import argparse

import torch
import torchvision.transforms as tvtrans

import glob 
import time

sys.path.insert(0, '../')

from lib.utils import load_model_weights,mkdir_p
from models.GALIP import NetG, CLIP_TXT_ENCODER

device = 'cuda' if torch.cuda.is_available() else 'cpu'




IMAGE_SIZE = 224

data = torch.load("../../data/captions_dict.pp")


def generate(args):
    # loading model parameters
    # clip
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model = clip_model.eval()

    # text encoder
    text_encoder = CLIP_TXT_ENCODER(clip_model).to(device)

    # netG
    netG = NetG(64, 100, 512, 256, 3, False, clip_model).to(device)
    path = '../saved_models/pretrained/pre_coco.pth'
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    netG = load_model_weights(netG, checkpoint['model']['netG'], multi_gpus=False)

    batch_size = 1
    noise = torch.randn((batch_size, 100)).to(device)


    # generating 
    num = 0
    total = len(data.keys())
    # generate from text
    with torch.no_grad():
        for key in tqdm(data.keys(),disable=True):
            num+=1
            start_time = time.time()

            caption_id = key
            caption=data[key]["caption"]
            image_id=data[key]["image_id"]
        
            tokenized_text = clip.tokenize([caption]).to(device)
            sent_emb, word_emb = text_encoder(tokenized_text)
            fake_imgs = netG(noise,sent_emb,eval=True).float()

            SAVE_PATH = F"{args.SAVE_PATH}/{IMAGE_SIZE}/{args.type}/{image_id}_{caption_id}.jpg"
            vutils.save_image(fake_imgs.data, SAVE_PATH, value_range=(-1, 1), normalize=True)

            end_time = time.time()
            if num % 1000==0:
                print(F"{num}/{total}\t{end_time-start_time:.2}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-T','--type', type=str, default='T2I',
                        help='batch size for parallel runs') # ["T2I","I2I","TI2I"]
    

    args = parser.parse_args()


    if socket.gethostname() =="PC": 
        args.SAVE_PATH =F"/home/xxx/work/datasets/AIGC_EVAL/AIGI/full/GALIP"

    else:
        args.SAVE_PATH = F"/home/special/user_new/xxx/datasets/COCO_2017/AIGC_EVAL/AIGI/full/GALIP"


    generate(args)
