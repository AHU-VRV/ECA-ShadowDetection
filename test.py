from image_loader import *
from torch.autograd import Variable
import tqdm
import cv2
from model import *
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def test():
    net = SHADOW().cuda()
    net.load_state_dict(torch.load('./model/ISTD.PTH'))
    net.eval()

    images_list = glob.glob('data/testISTD/*.*')

    for i, path in tqdm.tqdm(enumerate(images_list)):

        img = Image.open(path)

        W=img.size[0]
        H=img.size[1]
        img = Variable(transform(img).unsqueeze(0)).cuda()

        img= net(img)

        img =img.mul(255).byte()
        img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
        img = cv2.resize(img, (W, H))
        #img[img>=130]=255
        #img[img<=130]=0
        cv2.imwrite(os.path.join("BER/ISTD/myTest",os.path.basename(path)[:-4:]) +'.png',img)


test()
