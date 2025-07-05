
import pyiqa
import torch
# -*- coding:utf-8 -*-
import os
from PIL import Image
import numpy as np


from third import piqe as piqe_metric
import torch
# from torch.utils import data
from torchvision import transforms as T
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import socket
import glob
from torchvision import datasets
from torchvision.utils import save_image
import argparse
import random
import cv2
import time
import json
import torchvision.models as models
import multiprocessing
import PIL

from models.u2net_model import U2NET

from tqdm import tqdm

from cleanfid import fid


device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("cpu")


model_configs = {
    "GALIP": {
        "image_size": 224,
        "sub_sets": ["T2I"]
    },
    "DFGAN": {
        "image_size": 256,
        "sub_sets": ["T2I"]
    },
    "SD_V14": {
        "image_size": 512,
        "sub_sets": ["I2I", "T2I", "TI2I"]
    },
    "SD_V15": {
        "image_size": 512,
        "sub_sets": ["I2I", "T2I", "TI2I"]
    },
    "Versatile": {
        "image_size": 512,
        "sub_sets": ["I2I", "T2I", "TI2I"]
    },
    "SD_V21": {
        "image_size": 512,
        "sub_sets": ["I2I", "T2I", "TI2I"]
    },
    "SD_XL": {
        "image_size": 512,
        "sub_sets": ["I2I", "T2I", "TI2I"]
    },
    "Dalle2": {
        "image_size": 512,
        "sub_sets": ["T2I", "I2I"]
    },
    "Dalle3": {
        "image_size": 1024,
        "sub_sets": ["T2I"]
    },

    "T2I": ["GALIP", "DFGAN", "SD_V14", "SD_V15", "Versatile",
            "SD_V21", "SD_XL", "Dalle2", "Dalle3"],
    "I2I": ["SD_V14", "SD_V15", "Versatile",
            "SD_V21", "SD_XL", "Dalle2"],
    "TI2I": ["SD_V14", "SD_V15", "Versatile",
             "SD_V21", "SD_XL"],

}


def check_filename(image_name):
    while len(image_name) != 16:
        image_name = "0"+image_name

    return image_name


def piqe_eval(args):

    with torch.no_grad():
        scores = []
        _filenames = []
        for filename in args.filenames:
            _filenames.append(filename)
            image = cv2.imread(F"{args.prefix_path}/{filename}")

            if image.sum() <= 10:
                continue

            score = piqe_metric.piqe(image)
            # print(score,type(score))
            try:
                scores.append(round(score.item(), 2))
            except:
                scores.append(round(score, 2))

    return _filenames, scores


def get_filenames(args, sub_set):
    filenames = []
    captions = []
    for key in args.selected_keys:
        caption_id = key
        caption = args.data[key]["caption"]
        image_id = args.data[key]["image_id"]
        image_name = args.data[key]["image_name"]

        if sub_set == "org":
            filename = check_filename(image_name)
            filename = F"{args.COCO_path}/{filename}"
            filenames.append(filename)
        else:
            filename = F"{image_id}_{caption_id}.jpg"
            image_name = F"{args.AIGC_path}/{sub_set}/{filename}"
            filenames.append(image_name)

        captions.append(caption)

    return filenames, captions


def get_filenames_org(args):
    all_filenames = []
    part_filenames = []
    all_captions = []
    part_captions = []
    for key in args.selected_keys:
        # caption_id = key
        caption = args.data[key]["caption"]
        # image_id=data[key]["image_id"]
        image_name = args.data[key]["image_name"]

        # categorey_id=data[key]["categorey_id"]
        # categorey_name=data[key]["categorey_name"]
        supercategory = args.data[key]["supercategory"]

        if supercategory != args.supercategory:
            continue

        filename = check_filename(image_name)
        filename = F"{args.COCO_path}/{filename}"
        if filename not in args.org_filenames:
            continue

        # part
        if filename not in part_filenames:
            part_filenames.append(filename)
            part_captions.append(caption)

        all_filenames.append(filename)
        all_captions.append(caption)

    return part_filenames, all_filenames, part_captions, all_captions


class CocoDataset(Dataset):
    def __init__(self, img_path, filenames,  transform):
        self.transform = transform
        self.img_path = img_path

        self.image_names = filenames

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, n):

        image = PIL.Image.open(
            F"{self.img_path}/{self.image_names[n]}").convert('RGB')
        image_tensor = self.transform(image)

        return image_tensor, self.image_names[n]


class NR(object):
    def __init__(self, args, caption=False):
        self.args = args
        self.metric = args.indicator
        self.batch_size = args.batch_size

        self.filenames = args.filenames
        self.caption = caption

        #
        self.total_batch = int(len(self.filenames)/self.batch_size)

        # load model
        self.load_model()

        if self.caption:
            with open("data/filename_caption_dict.json") as f:
                self.name_cpations = json.load(f)

    def load_model(self):

        if self.metric in ['niqe', 'brisque', 'cnniqa', 'musiq', 'nima', 'dbcnn', 'maniqa', 'tres', 'hyperiqa', 'ilniqe', 'nrqm', 'pi', 'cnniqa', 'musiq', 'dbcnn', 'maniqa', 'clipiqa', 'entropy', 'topiq_nr', 'laion_aes', 'liqe', 'wadiqam_nr', 'qalign', 'unique', 'inception_score', 'clipscore', 'piqe']:
            self.evaluator = pyiqa.create_metric(self.metric).to(device)
        elif self.metric in ['qalign_qua', 'qalign_aes']:
            self.evaluator = pyiqa.create_metric('qalign').to(device)
        elif self.metric == 'classification':
            self.evaluator = models.vit_b_16(pretrained=True).to(device)
        elif self.metric == 'segmentation':
            self.evaluator = U2NET(3, 1)
            self.evaluator.load_state_dict(torch.load(
                "models/ckpts/u2net.pth"))
            self.evaluator.to(device)
            self.evaluator.eval()
        else:
            raise NotImplementedError

    def load_data(self):
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        testset = CocoDataset(self.args.prefix_path, self.filenames, transform)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=False, num_workers=max(1, multiprocessing.cpu_count() - 1))

        return test_loader

    def normPRED(self, d):
        ma = torch.max(d)
        mi = torch.min(d)

        dn = (d-mi)/(ma-mi)

        return dn

    def to_mask(self, mask):
        pred = self.normPRED(mask)

        mask = pred.squeeze().detach()

        bound = 0.1
        mask = mask.gt(bound)

        return mask

    def batch_like(self, batch, filenames, captions=None):
        if batch == self.total_batch:
            batch_filenames = filenames[batch*self.batch_size:]

        else:
            batch_filenames = filenames[batch *
                                        self.batch_size:(batch+1)*self.batch_size]

        batch_images = None
        for i in range(len(batch_filenames)):
            image = Image.open(
                F"{self.args.prefix_path}/{batch_filenames[i]}").convert("RGB")
            image = np.array(image)

            image = torch.tensor(image.transpose(2, 0, 1) / 255).unsqueeze(0)

            if batch_images is None:
                batch_images = image
            else:
                batch_images = torch.cat((batch_images, image), dim=0)

        _batch_captions = []
        if self.caption:
            for filename in batch_filenames:
                caption = self.name_cpations["25014"][filename]
                _batch_captions.append(caption)

        return batch_images, _batch_captions, batch_filenames

    def eval(self, device):
        scores = []
        _filenames = []
        for batch in tqdm(range(self.total_batch+1), disable=True):
            batch_images, _batch_captions, batch_filenames = self.batch_like(
                batch, self.filenames)

            if batch_images is None:
                break

            if self.metric in ['cnniqa', 'musiq', 'dbcnn', 'maniqa', 'clipiqa', 'tres', 'topiq_nr', 'laion_aes', 'wadiqam_nr', 'unique', 'nima', 'hyperiqa', 'nrqm', 'liqe']:
                batch_images = batch_images.float()

            if self.metric == "clipscore":
                score = self.evaluator(batch_images, caption_list=_batch_captions)
            elif self.metric == "qalign_qua":
                score = self.evaluator(batch_images, task_="quality")
            elif self.metric == "qalign_aes":
                score = self.evaluator(batch_images, task_="aesthetic")
            else:
                score = self.evaluator(batch_images)

            for i in range(len(score)):
                scores.append(round(score[i].item(), 4))
                _filenames.append(batch_filenames[i])

        return _filenames, scores

    def eval_classification(self,):
        test_loader = self.load_data()

        scores = []
        _filenames = []
        for i_batch, data in tqdm(enumerate(test_loader), disable=False):
            x = data[0].to(device)

            pred_x = self.evaluator(x)
            pred_y = pred_x.argmax(1)

            pred_label = pred_y.cpu().numpy().tolist()

            _filenames.extend(data[1])
            scores.extend(pred_label)

        return _filenames, scores

    def eval_segmentation(self,):
        test_loader = self.load_data()

        _filenames = []
        masks = []
        for i_batch, data in tqdm(enumerate(test_loader), disable=True):
            x = data[0].to(device)
            pred_masks = self.evaluator(x)

            for i in range(len(pred_masks)):
                mask = self.to_mask(pred_masks[i])

                mask = mask.cpu().unsqueeze(0).unsqueeze(0)

                if len(_filenames) == 0:
                    masks = mask
                else:
                    masks = torch.vstack((masks, mask))

                _filenames.append(data[1][i])

        return _filenames, {"filenames": _filenames, "masks": masks}


class FR(object):
    def __init__(self, args, caption=False):
        self.args = args
        self.metric = args.indicator
        self.batch_size = args.batch_size

        self.filenames = args.filenames
        self.caption = caption

        #
        self.total_batch = int(len(self.filenames)/self.batch_size)

        # load model
        if self.metric in ['ssim', 'psnr', 'lpips', 'dists', 'vif', 'vsi', 'fsim', 'mad', 'fid', 'inception_score']:
            self.load_model()

        if self.caption:
            with open("data/filename_caption_dict.json") as f:
                self.name_cpations = json.load(f)

    def load_model(self):
        if self.metric in ['ssim', 'psnr', 'lpips', 'dists', 'vif', 'vsi', 'fsim', 'mad', 'fid', 'niqe', 'brisque', 'cnniqa', 'musiq', 'nima', 'dbcnn', 'maniqa', 'tres', 'hyperiqa', 'ilniqe', 'nrqm', 'pi', 'cnniqa', 'musiq', 'dbcnn', 'maniqa', 'clipiqa', 'entropy', 'topiq_nr', 'laion_aes', 'liqe', 'wadiqam_nr', 'qalign', 'unique', 'inception_score', 'clipscore',]:
            self.evaluator = pyiqa.create_metric(self.metric).to(device)
        elif self.metric in ['qalign_qua', 'qalign_aes']:
            self.evaluator = pyiqa.create_metric('qalign').to(device)
        else:
            raise NotImplementedError

    def set_path(self, args):
        self.COCO_path = args.COCO_path
        self.AIGC_path = args.AIGC_path
        self.IMAGE_SIZE = args.IMAGE_SIZE

    def batch_like(self, batch, filenames, captions=None):
        if batch == self.total_batch:
            batch_filenames = filenames[batch*self.batch_size:]

        else:
            batch_filenames = filenames[batch *
                                        self.batch_size:(batch+1)*self.batch_size]

        batch_images = None
        for i in range(len(batch_filenames)):
            image = Image.open(
                F"{self.args.prefix_path}/{batch_filenames[i]}").convert("RGB")
            image = np.array(image)

            image = torch.tensor(image.transpose(2, 0, 1) / 255).unsqueeze(0)

            if batch_images is None:
                batch_images = image
            else:
                batch_images = torch.cat((batch_images, image), dim=0)

        _batch_captions = []
        if self.caption:
            for filename in batch_filenames:
                caption = self.name_cpations["25014"][filename]
                _batch_captions.append(caption)

        return batch_images, _batch_captions, batch_filenames

    def eval(self):
        results = {}

        if self.metric == "fid":
            scores = self.evaluator(self.COCO_path, self.AIGC_path,
                                    dataset_res=self.IMAGE_SIZE, batch_size=self.batch_size)

            scores = round(scores, 4)

            results["scores"] = scores
        elif self.metric == "inception_score":
            scores = self.evaluator(self.AIGC_path, batch_size=self.batch_size)
            print(scores)

            results["mean"] = round(scores['inception_score_mean'], 4)
            results["std"] = round(scores['inception_score_std'], 4)

        elif self.metric in ['ssim', 'psnr', 'lpips', 'dists', 'vif', 'vsi', 'fsim', 'mad']:

            filenames = []
            scores = []

            for i in range(len(self.filenames)):

                org_filename = F"{self.COCO_path}/{self.filenames[i]}"
                dest_filename = F"{self.AIGC_path}/{self.filenames[i]}"

                score = self.evaluator(org_filename, dest_filename)
                filenames.append(self.filenames[i])
                scores.append(round(score.item(), 4))

            results["filenames"] = filenames
            results["scores"] = scores

        else:
            raise NotImplementedError

        return results

    def clean_fid_eval(self):
        if self.metric == "clipfid":
            scores = fid.compute_fid(self.COCO_path, self.AIGC_path, mode="clean", model_name="clip_vit_b_32", batch_size=self.batch_size)
            scores = round(scores, 4)
        elif self.metric == "kid":
            scores = fid.compute_kid(self.COCO_path, self.AIGC_path, mode="clean",  batch_size=self.batch_size)
            scores = round(scores, 4)
        else:
            raise ValueError(F"Invalid metric: {self.metric}")

        results = {}
        results["scores"] = scores

        return results

    def swd_eval(self):
        _ = torch.manual_seed(123)
        from torchmetrics.image.kid import KernelInceptionDistance
        kid = KernelInceptionDistance(subset_size=32)

        filenames = glob.glob(F"{self.COCO_path}/*.jpg")

        imgs_dist1 = None
        imgs_dist2 = None
        for filename in filenames:
            basename = os.path.basename(filename)
            img_source = Image.open(F"{self.COCO_path}/{basename}").convert("RGB")
            img_source = torch.from_numpy(np.array(img_source)).permute(2, 0, 1).unsqueeze(0)

            img_dst = Image.open(F"{self.AIGC_path}/{basename}").convert("RGB")
            img_dst = torch.from_numpy(np.array(img_dst)).permute(2, 0, 1).unsqueeze(0)

            if imgs_dist1 is None:
                imgs_dist1 = img_source
                imgs_dist2 = img_dst
            else:
                imgs_dist1 = torch.cat((imgs_dist1, img_source), dim=0)
                imgs_dist2 = torch.cat((imgs_dist2, img_dst), dim=0)

        kid.update(imgs_dist1, real=True)
        kid.update(imgs_dist2, real=False)
        kid_mean, kid_std = kid.compute()

        results = {}
        results["mean"] = round(kid_mean.item(), 4)
        results["std"] = round(kid_std.item(), 4)

        return results


# fid
def fid_eval(args):
    iqa_fid = pyiqa.create_metric("fid", device=device)

    scores = iqa_fid(args.COCO_path, args.AIGC_path,
                     dataset_res=args.IMAGE_SIZE, batch_size=args.batch_size)
    scores = round(scores, 2)

    return scores


def eval_inception_score(args):

    indicator = "inception_score"

    iqa_metric = pyiqa.create_metric(indicator, device=device)
    score = iqa_metric(args.img_path, batch_size=args.batch_size)

    # print(score)

    return score


def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
