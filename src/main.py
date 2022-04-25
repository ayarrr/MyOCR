from src.net.db_detector import SegDetector
from src.net.backbones.resnet import resnet50
import cv2 as cv
import torchvision.transforms as transforms

def main():
    img_path = '../pic/img_264.jpg'
    img = cv.imread(img_path)
    f = transforms.ToTensor()(img)
    db = SegDetector()
    resnet = resnet50()
    res = db(resnet(f))
    print(res)



if __name__ == '__main__':
    main()


