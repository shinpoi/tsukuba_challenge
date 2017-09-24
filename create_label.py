# encode: utf-8
# python3
# create label for darknet

import numpy as np
import cv2
import os

# absoulute directoy
imgs_dir = "/home/shin-u16/document/tsukuba_img/dataset/JPEGImages/"
labels_npy = "/home/shin-u16/document/tsukuba_img/dataset/coordinate_array_01.npy"


# create name list
img_namelist = os.listdir(imgs_dir)
with open("namelist", 'w') as f:
    for name in img_namelist:
        f.write(imgs_dir + name + '\n')

# create labels
try:
    os.mkdir("labels")
except FileExistsError:
    pass


# images have same size:
img = cv2.imread(imgs_dir + '1.jpg')
x = float(img.shape[1])
y = float(img.shape[0])
l = np.load(labels_npy)
for ll in l:
    if ll[0] % 100 == 0:
        print("image: %d" % ll[0])
    # images have diffrent size:
    # img = cv2.imread(imgs_dir + str(ll[0]) + '.jpg')
    # x = img.shape[1]
    # y = img.shape[0]
    class_ = 14
    xc = (ll[4] + ll[2])*0.5/x
    yc = (ll[3] + ll[1])*0.5/y
    width = (ll[4] - ll[2])/x
    height = (ll[3] - ll[1])/y
    with open("labels/"+str(ll[0])+'.txt', 'w') as f:
        if ll[5] == 2:
            f.write("")
        else:
            f.write("%d %f %f %f %f" % (class_, xc, yc, width, height))


