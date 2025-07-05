import pyiqa
import torch
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torchvision import transforms as T
import glob

# list all available metrics
print(pyiqa.list_models())

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


class MyDataset(data.Dataset):
    def __init__(self, data_dir, transforms=None, train=True, imside=256):

        self.train = train

        self.imside = imside  # 128, 224

        self.data_dir = data_dir

        self.transforms = transforms

        if transforms is None:
            self.transforms = T.Compose([
                T.Resize([self.imside, self.imside]),
                T.ToTensor(),
            ])

        self._read_data()

    def _read_data(self):
        print(self.data_dir)
        self.data_names = glob.glob(F"{self.data_dir}/*.jpg")

    def __getitem__(self, index):
        # img_path = self.images[index]
        name = self.data_names[index]

        image = Image.open(name).convert("RGB")

        image = self.transforms(image)

        return image, name

    def __len__(self):
        return len(self.data_names)


def data_loader(args):
    dataset = MyDataset(data_dir=args.data_dir, train=True)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    return loader


class multidict(dict):
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value
