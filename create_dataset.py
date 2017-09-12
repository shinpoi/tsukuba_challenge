# - coding: utf-8 -*-
# python 3.5

import numpy as np
import cv2
import os
import random


#############################
#  para
SAMPLE = 2  # number of samples form every background


# input: filesname of humans, return a list of humans image's abs path
def get_human_list(filesname):
    list_ = []
    for i in os.walk(filesname):
        root, folder, files = i
        for file in files:
            dir_ = root + "/" + file
            list_.append(dir_)
    return list_


# input: files name, handle function
def ergodic_backgrounds(filesname, handle, hm=True, n_lim=float("inf"), save_root='gen'):
    for i in os.walk(filesname):
        root, folder, files = i
        for file in files:
            bk_ = root + "/" + file
            n = handle(bk_, hm=hm, save_root=save_root)
            if n % 100 == 0:
                print("n = %d" % n)
            if n >= n_lim:
                return None


# return a distorted imgArr
def human_distortion(imgArr, r=0.6, t=0.2):
    # size (0.6 ~ 0.8)
    times = round(r + random.random()*t, 2)
    imgArr = cv2.resize(imgArr, (int(imgArr.shape[1]*times), int(imgArr.shape[0]*times)))
    """
    # rotate (-20 ~ 20)
    angle = int(40*(random.random()-0.5))
    M = cv2.getRotationMatrix2D((imgArr.shape[0] / 2, imgArr.shape[1] / 2), angle, 1)
    max_ = max(imgArr.shape[0], imgArr.shape[1])
    imgArr = cv2.warpAffine(imgArr, M, (int(max_*1.5), int(max_*1.5)))
    # Perspective
    # pass
    """
    if random.random() > 0.5:
        imgArr = cv2.flip(imgArr, 1)
    return imgArr


def histogram_matching(srcArr, dstArr, srcPNG=True):
    src_HSV = cv2.cvtColor(srcArr, cv2.COLOR_RGB2HSV)
    srcHist = cv2.calcHist((src_HSV,), (2,), None, (256,), (0, 256)).reshape((-1,))
    if srcPNG:
        srcHist[0] = 0
    srcHist /= sum(srcHist)
    srcHistMap = np.zeros(256, dtype=np.float32)
    for i in range(len(srcHist)):
        srcHistMap[i] = sum(srcHist[:i])

    dst_HSV = cv2.cvtColor(dstArr, cv2.COLOR_RGB2HSV)
    dstHist = cv2.calcHist((dst_HSV,), (2,), None, (256,), (0, 256)).reshape((-1,))
    dstHist /= sum(dstHist)
    dstHistMap = np.zeros(256, dtype=np.float32)
    for i in range(len(dstHist)):
        dstHistMap[i] = sum(dstHist[:i])

    HistMap = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        minMap = 1
        minTag = None
        for j in range(256):
            if minMap > abs(srcHistMap[i] - dstHistMap[j]):
                minMap = abs(srcHistMap[i] - dstHistMap[j])
                minTag = j
        HistMap[i] = minTag
        # flatten??? may be...
        if i > 100000:
            if HistMap[i] < HistMap[i-1]:
                HistMap[i] = HistMap[i-1]
            if HistMap[i] == HistMap[i-1] == HistMap[i-2] == HistMap[i-3]:
                HistMap[i] += 1

    for i in range(src_HSV.shape[0]):
        for j in range(src_HSV.shape[1]):
            if src_HSV[i, j, 2] == 0:
                continue
            else:
                src_HSV[i, j, 2] = HistMap[src_HSV[i, j, 2]]

    return cv2.cvtColor(src_HSV, cv2.COLOR_HSV2RGB)


# input: full path of background.   combine and save image, save coordinate
def handle(bk_path, hm=True, save_root='gen'):
    humans = random.sample(human_list, SAMPLE)
    background = cv2.imread(bk_path)
    if background.shape[0] > 480:
        background = cv2.resize(background, (270, 480))
    bk_rows, bk_cols, ch = background.shape
    # read and distort human
    for human in humans:
        if 'blue' in human:
            color = 0
        elif 'orange' in human:
            color = 1
        else:
            raise ValueError("not found color!")
        bk = background.copy()
        rgba = cv2.imread(human, -1)
        if rgba.shape[0] < 200:
            rgba = human_distortion(rgba, r=1, t=0.5)
        elif rgba.shape[0] < 260:
            rgba = human_distortion(rgba, r=0.7, t=0.3)
        else:
            rgba = human_distortion(rgba)

        # histogram matching
        if hm:
            rgba[:, :, :3] = histogram_matching(rgba[:, :, :3], background)

        # insert coordinate
        hum_rows, hum_cols, ch = rgba.shape
        lim_rows = int((bk_rows - hum_rows)/2)
        lim_cols = bk_cols - hum_cols
        row_start = int(lim_rows*random.random()) + lim_rows
        col_start = int(lim_cols*random.random())

        # create mask
        mask = cv2.GaussianBlur(rgba[:, :, 3], (1, 1), 1)
        mask_inv = cv2.bitwise_not(mask)
        mask = np.array(mask, dtype=np.float32)/255
        mask_inv = np.array(mask_inv, dtype=np.float32)/255
        mask.resize((hum_rows, hum_cols, 1))
        mask_inv.resize((hum_rows, hum_cols, 1))
        mask = np.concatenate((mask, mask, mask), axis=2)
        mask_inv = np.concatenate((mask_inv, mask_inv, mask_inv), axis=2)

        # insert
        # print(row_start, col_start, hum_rows, hum_cols)
        bk_part = bk[row_start:row_start+hum_rows, col_start:col_start+hum_cols]
        bk[row_start:row_start + hum_rows, col_start:col_start + hum_cols] = \
            np.array(bk_part * mask_inv + rgba[:, :, :3] * mask, dtype=np.uint8)
        # save image and note coordinate
        # global coordinate_list
        global img_num
        global coordinate_array
        cv2.imwrite("%s/%d.jpg" % (save_root, img_num), bk)
        coordinate_array[img_num] = (img_num, row_start, col_start, row_start+hum_rows, col_start+hum_cols, color)
        img_num += 1

        if random.random() > 0.6:
            cv2.imwrite("%s/%d.jpg" % (save_root, img_num), background)
            coordinate_array[img_num] = (
            img_num, 0, 0, 0, 0, 2)
            img_num += 1

        return img_num - 1


#################################
try:
    os.mkdir("gen_01")
except FileExistsError:
    pass
try:
    os.mkdir("gen_02")
except FileExistsError:
    pass

lim = 140000
human_list = get_human_list("humans")

img_num = 0
# (img_number, start_y, start_x, end_y, end_x, color)
# color: blue-0, orange-1, none-2
coordinate_array = np.zeros((lim, 6), dtype=np.int32)
ergodic_backgrounds("backgrounds", handle, n_lim=lim, save_root='gen_01')
np.save("coordinate_array_01.npy", coordinate_array)


img_num = 0
coordinate_array = np.zeros((lim, 6), dtype=np.int32)
ergodic_backgrounds("backgrounds", handle, n_lim=lim, hm=False, save_root='gen_02')
np.save("coordinate_array_02.npy", coordinate_array)

