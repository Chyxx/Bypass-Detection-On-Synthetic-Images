import argparse
import os.path

import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from visdom import Visdom

import torch

from utils.utils import str2bool

from processor import Processor
from detector import Detector


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension")
parser.add_argument("--epsilon", type=float, default=5e-3, help="coefficient in the loss function")
parser.add_argument(
    "-f", "--file", default="E:\data", type=str, help="path to data directory"
)
parser.add_argument(
    "-m",
    "--model_path",
    type=str,
    default="data/detector-ckpt/imagenet_adm.pth",
)
parser.add_argument("--use_cpu", action="store_true", help="uses gpu by default, turn on to use cpu")
parser.add_argument("--arch", type=str, default="resnet50")
parser.add_argument("--aug_norm", type=str2bool, default=False)
parser.add_argument("--image_channels", type=int, default=3, help="number of image channels")
parser.add_argument("--model_channels", type=int, default=64, help="number of model channels")
parser.add_argument("--path", type=str, default="data/processor-ckpt/1_1800_.pkl")
opt = parser.parse_args()


transform = transforms.Compose([
    transforms.Resize(opt.img_size),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
test_data = torchvision.datasets.ImageFolder(root=os.path.join(opt.file, "sdv2-test"), transform=transform)
test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=True)


def norm(x):
    one = torch.ones_like(x)
    zero = torch.zeros_like(x)
    x = torch.where(x > 1, one, x)
    x = torch.where(x < 0, zero, x)
    return x


def main():
    checkpoint = torch.load(opt.path)
    p_net = Processor(opt).cuda()
    p_net.load_state_dict(checkpoint['model_state_dict'])
    p_net.eval()
    d_net = Detector(opt).cuda()
    d_net.eval()

    wind = Visdom()

    avg_prob1 = 0
    avg_prob2 = 0
    for i, (imgs, _) in enumerate(test_loader):
        with torch.no_grad():
            imgs = imgs.cuda()
            noise = p_net(imgs)
            p_imgs = norm(imgs + noise)
            prob1 = d_net(imgs)
            prob2 = d_net(p_imgs)
            loss = (prob2 / prob1).mean()

        print("i: {}\nprob1: {}, prob2: {}\nloss: {}".format(
            i, prob1.mean(), prob2.mean(), loss.item()))
        if i % 10 == 0:
            wind.images(imgs.cpu(), win="imgs")
            wind.images(p_imgs.cpu(), win="p_imgs")

        avg_prob1 += prob1.mean() * imgs.size(0) / len(test_data)
        avg_prob2 += prob2.mean() * imgs.size(0) / len(test_data)

    print("avg_prob1: {}, avg_prob2: {}".format(avg_prob1, avg_prob2))


if __name__ == "__main__":
    main()

