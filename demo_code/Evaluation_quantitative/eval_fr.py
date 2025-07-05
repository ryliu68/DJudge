import torch

import json

import argparse

from eval_utils import model_configs, FR, fix_seed


from util import multidict


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

BATH_PATH = "/home/xxx/work/datasets/COCO_2017/AIGC"


def fr_eval(args):

    args.results = multidict()

    results_path = F"results/fr/{args.indicator}.json"

    AIGCs = ["SD_V14", "SD_V15", "Versatile", "SD_V21", "SD_XL", "GALIP", "DFGAN", "Dalle2", "Dalle3"]

    if args.indicator in ['inception_score']:
        args.batch_size = 512
    elif args.indicator in ['lpips', 'dists']:
        args.batch_size = 512
    elif args.indicator in ['fid', 'clipfid', 'kid']:
        args.batch_size = 1024

    else:
        args.batch_size = 128
    print("Metric: %s" % args.indicator)

    # load model
    FReval = FR(args=args, caption=args.caption)

    for AIGC in AIGCs:

        args.IMAGE_SIZE = model_configs[AIGC]["image_size"]
        args.sub_sets = model_configs[AIGC]["sub_sets"]

        args.COCO_path = F"{BATH_PATH}/FID/ORG/{args.IMAGE_SIZE}"

        for sub_set in args.sub_sets:
            args.AIGC_path = F"{BATH_PATH}/FID/{AIGC}/{args.IMAGE_SIZE}/{sub_set}"

            FReval.set_path(args)

            if args.indicator in ['ssim', 'psnr', 'lpips', 'dists', 'vif', 'vsi', 'fsim', 'mad', 'fid', 'inception_score']:
                results = FReval.eval()
            elif args.indicator == "swd":
                results = FReval.swd_eval()
            elif args.indicator in ["clipfid", "kid"]:
                results = FReval.clean_fid_eval()
            else:
                raise ValueError(F"Invalid indicator: {args.indicator}")

            if "filenames" in results.keys() and "scores" in results.keys():
                args.results[F"{str(args.image_num)}"][AIGC][args.indicator][sub_set] = dict(zip(results["filenames"], results["scores"]))

            elif "filenames" not in results.keys() and "scores" in results.keys():
                args.results[F"{str(args.image_num)}"][AIGC][args.indicator][sub_set] = results["scores"]

            elif "mean" in results.keys() and "std" in results.keys():
                args.results[F"{str(args.image_num)}"][AIGC][args.indicator][sub_set] = {"mean": results["mean"], "std": results["std"]}

            else:
                raise ValueError

        with open(results_path, 'w') as json_file:
            json.dump(args.results, json_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-B', "--batch_size", type=int, default=64,
                        help='batch size for dataloader')
    parser.add_argument('-C', "--caption", type=str, default=False)
    parser.add_argument('-I', "--indicator", type=str, required=True, choices=['ssim', 'psnr', 'lpips', 'dists', 'vif', 'vsi', 'fsim', 'mad', "fid", "clipfid", "kid", "swd", 'inception_score'],
                        help='batch size for dataloader')
    args = parser.parse_args()

    fix_seed(2024)

    args.image_filenames_json = "data/image_filename-caption.json"

    # read json
    with open(args.image_filenames_json, 'r') as file:
        filenames = json.load(file)

    args.filenames = filenames["filenames"]

    if args.indicator in ['ssim', 'psnr', 'lpips', 'dists', 'vif', 'vsi', 'fsim', 'mad', "fid", "clipfid", "kid", "swd", 'inception_score']:
        fr_eval(args)
    else:
        raise ValueError
