import os, sys
import os.path as osp
import time
import random
import argparse
import numpy as np
from PIL import Image
import pprint
from tqdm import tqdm

import socket
import torch
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image, make_grid
import torchvision.utils as vutils
import multiprocessing as mp
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)
from lib.utils import mkdir_p, get_rank, merge_args_yaml, get_time_stamp, load_netG
from lib.utils import  truncated_noise, prepare_sample_data,get_tokenizer
from lib.perpare import prepare_models



def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='DF-GAN')
    parser.add_argument('--cfg', dest='cfg_file', type=str, default='./cfg/coco.yml',
                        help='optional config file')
    parser.add_argument('-T','--type', type=str, default='T2I',
                        help='batch size for parallel runs') # ["T2I","I2I","TI2I"]
    parser.add_argument('--imgs_per_sent', type=int, default=1,
                        help='the number of images per sentence')
    parser.add_argument('--imsize', type=int, default=256,
                        help='image szie')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='if use GPU')
    parser.add_argument('--train', type=bool, default=False,
                        help='if training')
    parser.add_argument('--multi_gpus', type=bool, default=False,
                        help='if use multi-gpu')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='gpu id')
    parser.add_argument('--local_rank', default=-1, type=int,
        help='node rank for distributed training')
    parser.add_argument('--random_sample', action='store_true',default=True, 
        help='whether to sample the dataset with random sampler')
    args = parser.parse_args()
    return args


def tokenize(wordtoix, text_filepath):
    '''generate images from example sentences'''
    tokenizer = get_tokenizer()
    data = torch.load("../../data/captions_dict.pp")

    captions = []
    cap_lens = []
    image_names = []
    for key in tqdm(data.keys(),disable=True):
        caption_id = key
        caption=data[key]["caption"]
        image_id=data[key]["image_id"]

        image_name = F"{image_id}_{caption_id}.jpg"
        image_names.append(image_name)

        # if len(image_names)<=10:
        #     print(caption)

        sent = caption.replace("\ufffd\ufffd", " ")
        tokens = tokenizer.tokenize(sent.lower())
        if len(tokens) == 0:
            print('sent', sent)
            continue
        rev = []
        for t in tokens:
            t = t.encode('ascii', 'ignore').decode('ascii')
            if len(t) > 0 and t in wordtoix:
                rev.append(wordtoix[t])
        captions.append(rev)
        cap_lens.append(len(rev))
    
    return captions, cap_lens,image_names

def build_word_dict(pickle_path):
    with open(pickle_path, 'rb') as f:
        x = pickle.load(f)
        wordtoix = x[3]
        del x
        n_words = len(wordtoix)
        print('Load from: ', pickle_path)
    return n_words, wordtoix


def sample_example(wordtoix, netG, text_encoder, args):
    captions, cap_lens,image_names = tokenize(wordtoix, args.example_captions)
    sent_embs, _  = prepare_sample_data(captions, cap_lens, text_encoder, args.device)
    
    caption_num = sent_embs.size(0)
    # get noise
    if args.truncation==True:
        noise = truncated_noise(args.imgs_per_sent, args.z_dim, args.trunc_rate)
        noise = torch.tensor(noise, dtype=torch.float).to(args.device)
    else:
        noise = torch.randn(args.imgs_per_sent, args.z_dim).to(args.device)
    # sampling
    with torch.no_grad():
        for i in tqdm(range(caption_num),disable=True):
            start_time = time.time()

            sent_emb = sent_embs[i].unsqueeze(0)
            fakes = netG(noise, sent_emb)   
            SAVE_PATH = F"{args.SAVE_PATH}/{args.imsize}/{args.type}/{image_names[i]}"
            vutils.save_image(fakes.data, SAVE_PATH, nrow=4, value_range=(-1, 1), normalize=True)
            torch.cuda.empty_cache()

            end_time = time.time()
            if i % 1000==0:
                print(F"{i}/{caption_num}\t{end_time-start_time:.2}s")

            # if i >=10:
            #     break


def main(args):
    # prepare data
    pickle_path = os.path.join(args.data_dir, 'captions_DAMSM.pickle')
    args.vocab_size, wordtoix = build_word_dict(pickle_path)
    # prepare models
    _, text_encoder, netG, _, _ = prepare_models(args)
    model_path = osp.join(ROOT_PATH, args.checkpoint)
    netG = load_netG(netG, model_path, args.multi_gpus, train=False)
    netG.eval()
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        print('Load %s for NetG'%(args.checkpoint))
        print("************ Start sampling ************")
    start_t = time.time()
    sample_example(wordtoix, netG, text_encoder, args)
    end_t = time.time()
    if (args.multi_gpus==True) and (get_rank() != 0):
        None
    else:
        print('*'*40)
        print('Sampling done, %.2fs cost, saved to %s'%(end_t-start_t, args.SAVE_PATH))
        print('*'*40)


if __name__ == "__main__":
    args = merge_args_yaml(parse_args())
    # set seed
    if args.manual_seed is None:
        args.manual_seed = 100
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if args.cuda:
        if args.multi_gpus:
            torch.cuda.manual_seed_all(args.manual_seed)
            torch.distributed.init_process_group(backend="nccl")
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            args.device = torch.device("cuda", local_rank)
            args.local_rank = local_rank
        else:
            torch.cuda.manual_seed_all(args.manual_seed)
            torch.cuda.set_device(args.gpu_id)
            args.device = torch.device("cuda")
    else:
        args.device = torch.device('cpu')

    
    if socket.gethostname() =="PC": 
        args.SAVE_PATH =F"/home/xxx/work/datasets/AIGC_EVAL/AIGI/full/DF-GAN"

    else:
        args.SAVE_PATH = F"/home/special/user_new/xxx/datasets/COCO_2017/AIGC_EVAL/AIGI/full/DF-GAN"

    main(args)
