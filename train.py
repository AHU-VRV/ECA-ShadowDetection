from image_loader import *
from model import *
import torch
from torchvision import transforms
import joint_transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.backends import cudnn
import numpy as np


min=1600


cudnn.benchmark = True

torch.cuda.set_device(0)

joint_transform = joint_transforms.Compose([
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.Resize((256, 256))
])
val_joint_transform = joint_transforms.Compose([
    joint_transforms.Resize((256, 256))
])
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

train_set = ImageFolder(training_root, joint_transform, img_transform, target_transform)

train_loader = DataLoader(train_set, batch_size=8, num_workers=0, shuffle=True)


bce= nn.BCEWithLogitsLoss().cuda()
remove = torch.nn.L1Loss().cuda()


def wbce(pred, gt):
    pos = torch.eq(gt, 1).float()
    neg = torch.eq(gt, 0).float()
    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg
    alpha_pos = num_neg / num_total
    alpha_neg = num_pos / num_total
    weights = alpha_pos * pos + alpha_neg * neg
    return nn.functional.binary_cross_entropy_with_logits(pred, gt, weights)

def single_gpu_train():

    net = SHADOW().cuda().train()
    optimizerd = torch.optim.Adamax([{'params': net.parameters()}], lr=0.0005)
    optimizer = torch.optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * 5e-3},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': 5e-3, 'weight_decay': 5e-4}
    ], momentum=0.9)
    #net.load_state_dict(torch.load('./model/ISTD+/646.PTH'))


    for epoch in range(40):
        epoch_loss = 0

        for i, data in enumerate(train_loader):

            inputs, labels= data
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()


            optimizerd.zero_grad()

            out= net(inputs)

            loss= bce(out, labels)

            loss.backward()
            optimizerd.step()
            epoch_loss += loss.item()

            print('Epoch: %d |iter:%d| train loss: %.5f' % (epoch, i, loss))
        print(epoch_loss)
        global min
        print(min)
        if epoch_loss<=min:
            min=epoch_loss
            #torch.save(net.state_dict(), './model/model/sbu'+str(min)+'.PTH')
            torch.save(net.state_dict(), './model/ISTD.PTH')

        #result2txt = str(epoch_loss)
        #with open('./loss/SRDREMOVE.txt', 'a') as file_handle:
            #file_handle.write(result2txt)
            #file_handle.write('\n')



if __name__ == '__main__':

    single_gpu_train()
