from tensorflow import keras
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from dataloader import DataLoader
import glob
import cv2
import scipy
import numpy as np

parser = ArgumentParser()
parser.add_argument('--image_path', default='D:/MobWorx/VideoSupRes/Check_Finala/Img/', type=str, help="Path to the test image")

      
def main(img,count):
    args = parser.parse_args()
    model = keras.models.load_model('MobileSRNet.h5')
    input = keras.Input((None, None, 3))
    output = model(input)
    model = keras.models.Model(input, output)
    imgs_hr = []
    imgs_lr = []
    h, w = img_res = (1024,1024)
    low_h, low_w = int(h / 4), int(w / 4)

    img_hr = scipy.misc.imresize(img, img_res)
    img_lr = scipy.misc.imresize(img, (low_h, low_w))

    # If training => do random flip

    imgs_hr.append(img_hr)
    imgs_lr.append(img_lr)

    imgs_hr = np.array(imgs_hr) / 127.5 - 1.
    imgs_lr = np.array(imgs_lr) / 127.5 - 1.

    fake_hr = model.predict(imgs_lr)
    img_lr = imgs_lr
    img_hr = imgs_hr
	
    #fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].imshow(fake_hr[0] * 0.5 + 0.5)
    ax[1].imshow(img_lr[0] * 0.5 + 0.5)
    #ax[2].imshow(img_hr[0] * 0.5 + 0.5)
    ax[0].set_title('Generated Super Resolution')
    ax[1].set_title('Low Res Input')
    #ax[2].set_title('High Res Original')
    fig.savefig(str(count)+'.jpg')
    #plt.show()


if __name__ == '__main__':
    filenames = sorted(glob.glob("D:/MobWorx/VideoSupRes/Check_Finala/Img/*.jpg"))
    images = [cv2.imread(img) for img in filenames]
    count = 43
    for img in images:
      main(img,count)
      count = count+1
