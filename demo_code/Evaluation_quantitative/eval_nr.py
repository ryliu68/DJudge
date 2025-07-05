import torch

from torchvision.utils import save_image
import argparse
from util import multidict
import json
from tqdm import tqdm


from eval_utils import model_configs, NR
from eval_utils import fix_seed

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

BATH_PATH = "/home/xxx/work/datasets/COCO_2017/AIGC"


def nr_eval(args):
    AIGCs = ["ORG", "SD_V14", "SD_V15", "Versatile",
             "SD_V21", "SD_XL", "GALIP", "DFGAN", "Dalle2", "Dalle3"]

    args.nr_results = multidict()

    nr_results_path = F"results/nr/{args.indicator}.json"

    print("Metric: %s" % args.indicator)

    if args.indicator in ['tres', 'hyperiqa']:
        args.batch_size = 8
    elif args.indicator in ['nrqm', 'ilniqe']:
        args.batch_size = 16
    elif args.indicator in ['entropy', 'musiq', 'dbcnn', 'maniqa', 'cnniqa',  'topiq_nr', 'liqe', 'qalign_qua', 'qalign_aes', 'segmentation']:
        args.batch_size = 32
    elif args.indicator in ['brisque', 'classification']:
        args.batch_size = 64
    elif args.indicator in ['unique']:
        args.batch_size = 128
    elif args.indicator in ['clipscore', 'clipiqa']:
        args.batch_size = 256
        args.caption = True
    elif args.indicator in ['laion_aes', 'nima']:
        args.batch_size = 512
    else:
        args.batch_size = 256

    NReval = NR(args=args, caption=args.caption)

    for AIGC in AIGCs:

        if AIGC == "ORG":
            # ORG
            args.IMAGE_SIZE = 512
            args.prefix_path = F"{BATH_PATH}/FID/ORG/{args.IMAGE_SIZE}"

            if args.indicator == "classification":
                _filenames, scores = NReval.eval_classification()
            elif args.indicator == "segmentation":
                _filenames, masks = NReval.eval_segmentation()
                torch.save(masks, F"output/masks/{AIGC}.pth")
                continue
            else:
                _filenames, scores = NReval.eval(args)

            args.nr_results[args.image_num][args.indicator]["ORG"] = {
                "name_scores": dict(zip(_filenames, scores))}

            with open(F"results/nr/{args.indicator}_ORG.json", 'w') as json_file:
                json.dump(args.nr_results, json_file)
            # ORG

        else:
            args.IMAGE_SIZE = model_configs[AIGC]["image_size"]
            args.sub_sets = model_configs[AIGC]["sub_sets"]

            for sub_set in tqdm(args.sub_sets, desc=F"{AIGC}"):
                args.prefix_path = F"{BATH_PATH}/FID/{AIGC}/{args.IMAGE_SIZE}/{sub_set}"

                if args.indicator == "classification":
                    _filenames, scores = NReval.eval_classification()
                elif args.indicator == "segmentation":
                    _filenames, masks = NReval.eval_segmentation()
                    torch.save(masks, F"output/masks/{AIGC}-{sub_set}.pth")
                    continue
                else:
                    _filenames, scores = NReval.eval(args)

                args.nr_results[args.image_num][args.indicator][AIGC][sub_set] = {
                    "name_scores": dict(zip(_filenames, scores))}

            with open(nr_results_path, 'w') as json_file:
                json.dump(args.nr_results, json_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-b', "--batch_size", type=int, default=8,
                        help='batch size for dataloader')
    parser.add_argument('-C', "--caption", type=str, default=False)
    parser.add_argument('-I', "--indicator", type=str, required=True, choices=['cnniqa', 'brisque', 'hyperiqa', 'liqe', 'unique', 'tres',
                                                                               'nima', 'laion_aes', 'qalign_qua', 'qalign_aes', 'clipscore', 'laion_aes', 'inception_score', 'piqe', 'nrqm', 'dbcnn', 'musiq', 'ilniqe', 'clipiqa', 'classification', 'segmentation'], help='batch size for dataloader')
    args = parser.parse_args()

    fix_seed(2024)

    args.image_filenames_json = "data/image_filename-caption.json"

    # read json
    with open(args.image_filenames_json, 'r') as file:
        filenames = json.load(file)

    args.filenames = filenames["filenames"]

    if args.indicator in ['cnniqa', 'brisque',  'unique', 'tres', 'hyperiqa', 'liqe', 'tres', 'nima', 'laion_aes', 'qalign_qua', 'qalign_aes', 'clipscore', 'laion_aes', 'inception_score', 'piqe', 'nrqm', 'dbcnn', 'musiq', 'ilniqe', 'clipiqa', 'classification', 'segmentation']:
        nr_eval(args)
    else:
        raise NotImplementedError
