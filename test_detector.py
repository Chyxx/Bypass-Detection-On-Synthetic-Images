from detector import Detector
import torch
from torch.utils.data import DataLoader
from config import opt
from utils import dataset
seed = 23
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
def main():
    data = dataset.ProcessorDataset(opt.file, opt.img_size)
    train_loader = DataLoader(data, batch_size=opt.batch_size, shuffle=True)
    d_net = Detector(opt).cuda()
    d_net.eval()
    for _, imgs in enumerate(train_loader):
        a = []
        for i in range(0,100):
            for j in range(0,16):
                a.append(d_net(imgs.cuda()).cpu()[j].item())
        for i in range(0,16):
            arr = [a[x*16 + i] for x in range(0,100)]
            b = [x**2 for x in arr]
            print("avg",sum(arr)/100,'var',sum(b)/100 - (sum(arr)**2)/10000)
        break;

if __name__ == "__main__":
    main()