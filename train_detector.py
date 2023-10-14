from config import opt
from utils import one_hot
from torchvision.models import resnet50, ResNet50_Weights
from dataset import DetectorDataset

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
import visdom
import detector

seed = 23
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子


def main():
    total_train_step = 0
    epoch_size = 50
    # visdom
    wind = visdom.Visdom()
    wind.line([0.], [0], win="loss", opts=dict(title="loss"))
    wind.line([0.], [0], win="accuracy", opts=dict(title="accuracy"))
    # data
    train_data = DetectorDataset("E:/data/imagenet_ai_0424_sdv5/train", 200)
    test_data = DetectorDataset("E:/data/imagenet_ai_0424_sdv5/val", 200)
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=8)
    # model
    model = detector.DetectorToTrain(opt)
    model = model.cuda()
    # optimizer
    learning_rate = 0.001
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    # loss
    criterion = nn.CrossEntropyLoss().cuda()
    for epoch in range(epoch_size):
        print("-------第{}轮训练开始-------".format(epoch))
        model.train()
        # 训练步骤开始
        for data in train_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()

            targets = one_hot(targets.cuda(), 2)
            outputs = model(imgs)
            loss = criterion(outputs, targets)

            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step = total_train_step + 1

            if total_train_step % 100 == 0:
                print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))

        # 测试集
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            correct = 0
            for data in test_dataloader:
                imgs, targets = data
                imgs = imgs.cuda()
                t = targets.cuda()
                targets = one_hot(targets.cuda(), 2)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                total_test_loss += loss.item() / len(test_dataloader)
                pred = outputs.argmax(dim=1)
                correct += pred.eq(t).sum().float().item()
            print("测试：{}，Loss：{}, Accuracy: {}".format(epoch, total_test_loss,
                                                         correct / len(test_dataloader.dataset)))
            wind.line([total_test_loss], [epoch], update="append", opts=dict(title="loss"), win="loss")
            wind.line([correct / len(test_dataloader.dataset)], [epoch], update="append", opts=dict(title="accuracy"),
                      win="accuracy")

        model.save(optimizer, epoch, 0)


if __name__ == "__main__":
    main()
