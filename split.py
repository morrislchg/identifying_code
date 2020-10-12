import numpy as np
import cv2
import os
import datetime as dt

# 将验证码分割为单个字符并对字符进行标注和分类


class Img:

    def __init__(self, dir_path):
        self.dir_path = dir_path

    def split_image(self, path):
        img = cv2.imread(path, 0)
        # 把图片变为二值图片
        ret, img1 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)
        # 分割图像
        num_list = []
        index_list = [0]
        for i in range(0, img1.shape[1]):
            num = cv2.countNonZero(img1[:, i: i + 1])
            num_list.append(num)
            # 寻找分割点
            if i > 0 and num_list[i - 1] != 0 and num_list[i] == 0:
                index_list.append(i + 1)

        # 分割图像并标注字符，将标注的字符分类储存在不同文件夹中
        for i in range(1, len(index_list)):
            img2 = img1[:, index_list[i-1]:index_list[i]]
            res = cv2.resize(img2, (30, 30), interpolation=cv2.INTER_CUBIC)
            cv2.imshow('image', res)
            k = cv2.waitKey()
            ch = chr(k)
            time = dt.datetime.now()
            time = time.strftime("%H_%M_%S")
            dirs = str(ch)
            # 创建新的文件夹
            if not os.path.exists(dirs):
                os.makedirs(dirs)

            img_path = dirs + '/train/' + str(time) + '_' + dirs + '.jpg'
            print(img_path)
            cv2.imwrite(img_path, res)

    def dispose_all_image(self):
        for i in range(0, 5):
            files = os.listdir(self.dir_path + '/' + str(i))
            for file in files:
                path = self.dir_path + '/' + str(i) + '/' + file
                self.split_image(path)


dir_path1 = '0424'
im = Img(dir_path1)
im.dispose_all_image()
