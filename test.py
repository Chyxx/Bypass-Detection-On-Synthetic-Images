import pytorch_msssim as PM
import torch
from torch.utils.data import DataLoader

from config import opt
from detector import Detector
from processor import Processor
from utils import dataset
from utils.utils import norm, get_prob


def main():

    data = dataset.ProcessorDataset(opt.file, opt.img_size)
    data_loader = DataLoader(data, batch_size=opt.batch_size, shuffle=False)
    p_net = Processor(opt).cuda().eval()
    d_net = Detector(opt).cuda().eval()

    with torch.no_grad():
        avg_prob1 = 0
        avg_prob2 = 0
        avg_ssim = 0
        correct1 = 0
        correct2 = 0
        for i, imgs in enumerate(data_loader):
            imgs = imgs.cuda()
            noise = p_net(imgs)
            p_imgs = norm(imgs + noise)
            prob1 = get_prob(d_net(imgs))
            prob2 = get_prob(d_net(p_imgs))
            ssim = PM.ssim(p_imgs, imgs)
            avg_prob1 += prob1.mean() * imgs.size(0) / len(data)
            avg_prob2 += prob2.mean() * imgs.size(0) / len(data)
            avg_ssim += ssim / len(data_loader)
            pred1 = torch.where(prob1 >= 0.5, 1, 0)
            pred2 = torch.where(prob2 >= 0.5, 1, 0)
            correct1 += pred1.sum().item()
            correct2 += pred2.sum().item()
            if i % 20 == 0:
                print("batch:{}, prob1:{}, prob2:{}".format(i, prob1.mean(), prob2.mean()))
        # print
        print("-------测试结束-------")
        print("avg_prob1:{}, avg_prob2:{}, avg_prob2/prob1：{}".format(avg_prob1, avg_prob2, avg_prob2 / avg_prob1))
        print("accuracy1:{}, accuracy2:{}".format(correct1/len(data), correct2/len(data)))
        print("avg_ssim:{}".format(avg_ssim))


if __name__ == "__main__":
    main()