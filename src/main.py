from src.net.db_detector import SegDetector
from src.net.backbones.resnet import resnet50,deformable_resnet50
import cv2
import torchvision.transforms as transforms
import math
from PIL import Image



def resize_image(img):
    height, width, _ = img.shape
    if height < width:
        new_height = 736
        new_width = int(math.ceil(new_height / height * width / 32) * 32)
    else:
        new_width = 736
        new_height = int(math.ceil(new_width / width * height / 32) * 32)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img


def main():
    img_path = '../pic/img_264.jpg'
    img = resize_image(cv2.imread(img_path).astype('float32'))
    f = transforms.ToTensor()(img)
    f = f.float().unsqueeze(0)
    db = SegDetector(in_channels=[256, 512, 1024, 2048],k=50,adaptive=True,bias=True)
    resnet = resnet50()
    res = db(resnet(f))
    # print(res)
    # print(res['binary'].shape)
    map_b = transforms.ToPILImage()(res['binary'][0]-1)
    map_t = transforms.ToPILImage()(res['thresh'][0])
    map_tb= transforms.ToPILImage()(res['thresh_binary'][0])
    Image.open(img_path).show()
    map_b.show()




if __name__ == '__main__':
    main()


