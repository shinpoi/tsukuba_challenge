# utf-8
# python3

import cv2
import numpy
import os

label_root = "./labels/"
image_root = "./images/"

try:
    os.mkdir("img_test")
except FileExistsError:
    pass

label_names = os.listdir(label_root)

for label in label_names:
    # read lable
    print("read %s" % label)
    l = []
    with open(label_root + label) as f:
        while True:
            a = f.readline()
            if not a:
                break
            else:
                l.append([float(i) for i in a.split(" ")])
    # read image
    img = cv2.imread(image_root + label[:-3] + "jpg")
    for ll in l:
        start_y = int((ll[2] - ll[4]/2) * img.shape[0])
        start_x = int((ll[1] - ll[3]/2) * img.shape[1])
        end_y = int((ll[2] + ll[4]/2) * img.shape[0])
        end_x = int((ll[1] + ll[3]/2) * img.shape[1])
        img[start_y, start_x:end_x] = (255, 255, 255)
        img[end_y, start_x:end_x] = (255, 255, 255)
        img[start_y: end_y, start_x] = (255, 255, 255)
        img[start_y: end_y, end_x] = (255, 255, 255)
        cv2.putText(img, str(int(ll[0])), (start_x, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # write image
    cv2.imwrite("img_test/"+label[:-3] + "jpg", img)
