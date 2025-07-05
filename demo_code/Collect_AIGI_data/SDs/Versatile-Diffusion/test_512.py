
import os
import PIL

import numpy as np


from tqdm import tqdm
import glob
import socket
import argparse

import torch
import torchvision.transforms as tvtrans
from lib.cfg_helper import model_cfg_bank
from lib.model_zoo import get_model

import glob 
import time

import os

n_sample_image = 1
n_sample_text = 4
cache_examples = True

from lib.model_zoo.ddim import DDIMSampler


def highlight_print(info):
    print('')
    print(''.join(['#']*(len(info)+4)))
    print('# '+info+' #')
    print(''.join(['#']*(len(info)+4)))
    print('')

def decompose(x, q=20, niter=100):
    x_mean = x.mean(-1, keepdim=True)
    x_input = x - x_mean
    u, s, v = torch.pca_lowrank(x_input, q=q, center=False, niter=niter)
    ss = torch.stack([torch.diag(si) for si in s])
    x_lowrank = torch.bmm(torch.bmm(u, ss), torch.permute(v, [0, 2, 1]))
    x_remain = x_input - x_lowrank
    return u, s, v, x_mean, x_remain

class adjust_rank(object):
    def __init__(self, max_drop_rank=[1, 5], q=20):
        self.max_semantic_drop_rank = max_drop_rank[0]
        self.max_style_drop_rank = max_drop_rank[1]
        self.q = q

        def t2y0_semf_wrapper(t0, y00, t1, y01):
            return lambda t: (np.exp((t-0.5)*2)-t0)/(t1-t0)*(y01-y00)+y00
        t0, y00 = np.exp((0  -0.5)*2), -self.max_semantic_drop_rank
        t1, y01 = np.exp((0.5-0.5)*2), 1
        self.t2y0_semf = t2y0_semf_wrapper(t0, y00, t1, y01)

        def x2y_semf_wrapper(x0, x1, y1):
            return lambda x, y0: (x-x0)/(x1-x0)*(y1-y0)+y0
        x0 = 0
        x1, y1 = self.max_semantic_drop_rank+1, 1
        self.x2y_semf = x2y_semf_wrapper(x0, x1, y1)
        
        def t2y0_styf_wrapper(t0, y00, t1, y01):
            return lambda t: (np.exp((t-0.5)*2)-t0)/(t1-t0)*(y01-y00)+y00
        t0, y00 = np.exp((1  -0.5)*2), -(q-self.max_style_drop_rank)
        t1, y01 = np.exp((0.5-0.5)*2), 1
        self.t2y0_styf = t2y0_styf_wrapper(t0, y00, t1, y01)

        def x2y_styf_wrapper(x0, x1, y1):
            return lambda x, y0: (x-x0)/(x1-x0)*(y1-y0)+y0
        x0 = q-1
        x1, y1 = self.max_style_drop_rank-1, 1
        self.x2y_styf = x2y_styf_wrapper(x0, x1, y1)

    def __call__(self, x, lvl):
        if lvl == 0.5:
            return x

        if x.dtype == torch.float16:
            fp16 = True
            x = x.float()
        else:
            fp16 = False
        std_save = x.std(axis=[-2, -1])

        u, s, v, x_mean, x_remain = decompose(x, q=self.q)

        if lvl < 0.5:
            assert lvl>=0
            for xi in range(0, self.max_semantic_drop_rank+1):
                y0 = self.t2y0_semf(lvl)
                yi = self.x2y_semf(xi, y0)
                yi = 0 if yi<0 else yi
                s[:, xi] *= yi

        elif lvl > 0.5:
            assert lvl <= 1
            for xi in range(self.max_style_drop_rank, self.q):
                y0 = self.t2y0_styf(lvl)
                yi = self.x2y_styf(xi, y0)
                yi = 0 if yi<0 else yi
                s[:, xi] *= yi
            x_remain = 0

        ss = torch.stack([torch.diag(si) for si in s])
        x_lowrank = torch.bmm(torch.bmm(u, ss), torch.permute(v, [0, 2, 1]))
        x_new = x_lowrank + x_mean + x_remain

        std_new = x_new.std(axis=[-2, -1])
        x_new = x_new / std_new * std_save

        if fp16:
            x_new = x_new.half()

        return x_new


class vd_inference(object):
    def __init__(self, fp16=False, which='v2.0'):
        highlight_print(which)
        self.which = which

        if self.which == 'v1.0':
            cfgm = model_cfg_bank()('vd_four_flow_v1-0')
        else:
            assert False, 'Model type not supported'
        net = get_model()(cfgm)

        if fp16:
            highlight_print('Running in FP16')
            if self.which == 'v1.0':
                net.ctx['text'].fp16 = True
                net.ctx['image'].fp16 = True
            net = net.half()
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        if self.which == 'v1.0':
            if fp16:
                sd = torch.load('/home/special/user_new/xxx/pretrained/Versatile-Diffusion/pretrained/vd-four-flow-v1-0-fp16.pth', map_location='cpu')
            else:
                sd = torch.load('/home/special/user_new/xxx/pretrained/Versatile-Diffusion/pretrained/vd-four-flow-v1-0.pth', map_location='cpu')

        net.load_state_dict(sd, strict=False)

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            net.to('cuda')
        self.net = net
        self.sampler = DDIMSampler(net)

        self.output_dim = [512, 512]
        self.n_sample_image = n_sample_image
        self.n_sample_text = n_sample_text
        self.ddim_steps = 50
        self.ddim_eta = 0.0
        self.scale_textto = 7.5
        self.image_latent_dim = 4
        self.text_latent_dim = 768
        self.text_temperature = 1

        if which == 'v1.0':
            self.adjust_rank_f = adjust_rank(max_drop_rank=[1, 5], q=20)
            self.scale_imgto = 7.5
            self.disentanglement_noglobal = True

    def inference_t2i(self, text, seed):
        n_samples = self.n_sample_image
        scale = self.scale_textto
        sampler = self.sampler
        h, w = self.output_dim
        u = self.net.ctx_encode([""], which='text').repeat(n_samples, 1, 1)
        c = self.net.ctx_encode([text], which='text').repeat(n_samples, 1, 1)
        shape = [n_samples, self.image_latent_dim, h//8, w//8]
        np.random.seed(seed)
        torch.manual_seed(seed + 100)
        x, _ = sampler.sample(
            steps=self.ddim_steps,
            x_info={'type':'image'},
            c_info={'type':'text', 'conditioning':c, 'unconditional_conditioning':u, 
                    'unconditional_guidance_scale':scale},
            shape=shape,
            verbose=False,
            eta=self.ddim_eta)
        im = self.net.vae_decode(x, which='image')
        im = [tvtrans.ToPILImage()(i) for i in im]
        return im

    def inference_i2i(self, im, fid_lvl, fcs_lvl, clr_adj, seed):
        n_samples = self.n_sample_image
        scale = self.scale_imgto
        sampler = self.sampler
        h, w = self.output_dim
        device = self.net.device

        BICUBIC = PIL.Image.Resampling.BICUBIC
        im = im.resize([w, h], resample=BICUBIC)

        if fid_lvl == 1:
            return [im]*n_samples

        cx = tvtrans.ToTensor()(im)[None].to(device).to(self.dtype)

        c = self.net.ctx_encode(cx, which='image')
        if self.disentanglement_noglobal:
            c_glb = c[:, 0:1]
            c_loc = c[:, 1: ]
            c_loc = self.adjust_rank_f(c_loc, fcs_lvl)
            c = torch.cat([c_glb, c_loc], dim=1).repeat(n_samples, 1, 1)
        else:
            c = self.adjust_rank_f(c, fcs_lvl).repeat(n_samples, 1, 1)
        u = torch.zeros_like(c)

        shape = [n_samples, self.image_latent_dim, h//8, w//8]
        np.random.seed(seed)
        torch.manual_seed(seed + 100)
        if fid_lvl!=0:
            x0 = self.net.vae_encode(cx, which='image').repeat(n_samples, 1, 1, 1)
            step = int(self.ddim_steps * (1-fid_lvl))
            x, _ = sampler.sample(
                steps=self.ddim_steps,
                x_info={'type':'image', 'x0':x0, 'x0_forward_timesteps':step},
                c_info={'type':'image', 'conditioning':c, 'unconditional_conditioning':u, 
                        'unconditional_guidance_scale':scale},
                shape=shape,
                verbose=False,
                eta=self.ddim_eta)
        else:
            x, _ = sampler.sample(
                steps=self.ddim_steps,
                x_info={'type':'image',},
                c_info={'type':'image', 'conditioning':c, 'unconditional_conditioning':u, 
                        'unconditional_guidance_scale':scale},
                shape=shape,
                verbose=False,
                eta=self.ddim_eta)

        imout = self.net.vae_decode(x, which='image')

        if clr_adj == 'Simple':
            cx_mean = cx.view(3, -1).mean(-1)[:, None, None]
            cx_std  = cx.view(3, -1).std(-1)[:, None, None]
            imout_mean = [imouti.view(3, -1).mean(-1)[:, None, None] for imouti in imout]
            imout_std  = [imouti.view(3, -1).std(-1)[:, None, None] for imouti in imout]
            imout = [(ii-mi)/si*cx_std+cx_mean for ii, mi, si in zip(imout, imout_mean, imout_std)]
            imout = [torch.clamp(ii, 0, 1) for ii in imout]

        imout = [tvtrans.ToPILImage()(i) for i in imout]
        return imout

    

    def inference_dcg(self, imctx, fcs_lvl, textctx, textstrength, seed):
        n_samples = self.n_sample_image
        sampler = self.sampler
        h, w = self.output_dim
        device = self.net.device

        c_info_list = []

        if (textctx is not None) and (textctx != "") and (textstrength != 0):
            ut = self.net.ctx_encode([""], which='text').repeat(n_samples, 1, 1)
            ct = self.net.ctx_encode([textctx], which='text').repeat(n_samples, 1, 1)
            scale = self.scale_imgto*(1-textstrength) + self.scale_textto*textstrength

            c_info_list.append({
                'type':'text', 
                'conditioning':ct, 
                'unconditional_conditioning':ut,
                'unconditional_guidance_scale':scale,
                'ratio': textstrength, })
        else:
            scale = self.scale_imgto
            textstrength = 0

        BICUBIC = PIL.Image.Resampling.BICUBIC
        cx = imctx.resize([w, h], resample=BICUBIC)
        cx = tvtrans.ToTensor()(cx)[None].to(device).to(self.dtype)
        ci = self.net.ctx_encode(cx, which='image')

        if self.disentanglement_noglobal:
            ci_glb = ci[:, 0:1]
            ci_loc = ci[:, 1: ]
            ci_loc = self.adjust_rank_f(ci_loc, fcs_lvl)
            ci = torch.cat([ci_glb, ci_loc], dim=1).repeat(n_samples, 1, 1)
        else:
            ci = self.adjust_rank_f(ci, fcs_lvl).repeat(n_samples, 1, 1)

        c_info_list.append({
            'type':'image', 
            'conditioning':ci, 
            'unconditional_conditioning':torch.zeros_like(ci),
            'unconditional_guidance_scale':scale,
            'ratio': (1-textstrength), })

        shape = [n_samples, self.image_latent_dim, h//8, w//8]
        np.random.seed(seed)
        torch.manual_seed(seed + 100)
        x, _ = sampler.sample_multicontext(
            steps=self.ddim_steps,
            x_info={'type':'image',},
            c_info_list=c_info_list,
            shape=shape,
            verbose=False,
            eta=self.ddim_eta)

        imout = self.net.vae_decode(x, which='image')
        imout = [tvtrans.ToPILImage()(i) for i in imout]
        return imout

    def inference_tcg(self, *args):
        args_imag = list(args[0:10]) + [None, None, None, None, None]*2
        args_rest = args[10:]
        imin, imout = self.inference_mcg(*args_imag, *args_rest)
        return imin, imout

    def inference_mcg(self, *args):
        imctx = [args[0:5], args[5:10], args[10:15], args[15:20]]
        textctx, textstrength, seed = args[20:]

        n_samples = self.n_sample_image
        sampler = self.sampler
        h, w = self.output_dim
        device = self.net.device

        c_info_list = []

        if (textctx is not None) and (textctx != "") and (textstrength != 0):
            ut = self.net.ctx_encode([""], which='text').repeat(n_samples, 1, 1)
            ct = self.net.ctx_encode([textctx], which='text').repeat(n_samples, 1, 1)
            scale = self.scale_imgto*(1-textstrength) + self.scale_textto*textstrength

            c_info_list.append({
                'type':'text', 
                'conditioning':ct, 
                'unconditional_conditioning':ut,
                'unconditional_guidance_scale':scale,
                'ratio': textstrength, })
        else:
            scale = self.scale_imgto
            textstrength = 0

        input_save = []
        imc = []
        for im, imm, strength, fcs_lvl, use_mask in imctx:
            if (im is None) and (imm is None):
                continue
            BILINEAR = PIL.Image.Resampling.BILINEAR
            BICUBIC = PIL.Image.Resampling.BICUBIC
            if use_mask:
                cx = imm['image'].resize([w, h], resample=BICUBIC)
                cx = tvtrans.ToTensor()(cx)[None].to(self.dtype).to(device)
                m = imm['mask'].resize([w, h], resample=BILINEAR)
                m = tvtrans.ToTensor()(m)[None, 0:1].to(self.dtype).to(device)
                m = (1-m)
                cx_show = cx*m
                ci = self.net.ctx_encode(cx, which='image', masks=m)
            else:
                cx = im.resize([w, h], resample=BICUBIC)
                cx = tvtrans.ToTensor()(cx)[None].to(self.dtype).to(device)
                ci = self.net.ctx_encode(cx, which='image')
                cx_show = cx

            input_save.append(tvtrans.ToPILImage()(cx_show[0]))

            if self.disentanglement_noglobal:
                ci_glb = ci[:, 0:1]
                ci_loc = ci[:, 1: ]
                ci_loc = self.adjust_rank_f(ci_loc, fcs_lvl)
                ci = torch.cat([ci_glb, ci_loc], dim=1).repeat(n_samples, 1, 1)
            else:
                ci = self.adjust_rank_f(ci, fcs_lvl).repeat(n_samples, 1, 1)
            imc.append(ci * strength)

        cis = torch.cat(imc, dim=1)
        c_info_list.append({
            'type':'image', 
            'conditioning':cis, 
            'unconditional_conditioning':torch.zeros_like(cis),
            'unconditional_guidance_scale':scale,
            'ratio': (1-textstrength), })

        shape = [n_samples, self.image_latent_dim, h//8, w//8]
        np.random.seed(seed)
        torch.manual_seed(seed + 100)
        x, _ = sampler.sample_multicontext(
            steps=self.ddim_steps,
            x_info={'type':'image',},
            c_info_list=c_info_list,
            shape=shape,
            verbose=False,
            eta=self.ddim_eta)

        imout = self.net.vae_decode(x, which='image')
        imout = [tvtrans.ToPILImage()(i) for i in imout]
        return input_save, imout


vd_inference = vd_inference(which='v1.0', fp16=True)



EXIST_PATH="/home/xxx/work/datasets/AIGC_EVAL/AIGI/Versatile/512/I2I"

IMAGE_SIZE = 512


if socket.gethostname() =="PC": 
    IMAGE_PATH ="/home/xxx/work/datasets/COCO_2017/val2017_resize/512"
    SAVE_PATH =F"/home/xxx/work/datasets/AIGC_EVAL/AIGI/full/Versatile"

else:
    IMAGE_PATH = "/home/special/user_new/xxx/datasets/COCO_2017/val2017_resize/512"
    SAVE_PATH = F"/home/special/user_new/xxx/datasets/COCO_2017/AIGC_EVAL/AIGI/full/Versatile"


_filenames = glob.glob(F"{EXIST_PATH}/*.*")

filenames = [item.split("/")[-1] for item in _filenames]



data = torch.load("../../data/captions_dict.pp")


def generate(args):
    num = 0
    total = len(data.keys())

    for key in tqdm(data.keys(),disable=True):
        num+=1
        start_time = time.time()


        caption_id = key
        caption=data[key]["caption"]
        image_id=data[key]["image_id"]
        image_name=data[key]["image_name"]

        while len(image_name)!=16:
            image_name="0"+image_name



        image_path = F"{IMAGE_PATH}/{image_name}"
        # print(image_path,caption)

        new_filename = F"{image_id}_{caption_id}.jpg"

        

        if args.type == "T2I":
            image = vd_inference.inference_t2i(text=caption,seed=20)[0]
            image.save(F"{SAVE_PATH}/{IMAGE_SIZE}/{args.type}/{new_filename}")
            
        elif args.type == "I2I":
            img_input=PIL.Image.open(image_path).convert("RGB")
            image=vd_inference.inference_i2i(img_input, fid_lvl=0.5, fcs_lvl=0.5, clr_adj=None, seed=20)[0]
            image.save(F"{SAVE_PATH}/{IMAGE_SIZE}/{args.type}/{new_filename}")


        elif args.type == "TI2I":
            # inference_dcg(self, imctx, fcs_lvl, textctx, textstrength, seed)
            img_input=PIL.Image.open(image_path).convert("RGB")

            image = vd_inference.inference_dcg(imctx=img_input,fcs_lvl=0.5,textctx=caption,textstrength=0.75,seed=20)[0]
            image.save(F"{SAVE_PATH}/{IMAGE_SIZE}/{args.type}/{new_filename}")
        else:
            raise ValueError(f"{args.type} is not a valid item")
            
        end_time = time.time()
        print(F"{num}/{total}\t{end_time-start_time:.2}s")
       

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-T','--type', type=str, default='T2I',
                        help='batch size for parallel runs') # ["T2I","I2I","TI2I"]
    

    args = parser.parse_args()

    generate(args)
