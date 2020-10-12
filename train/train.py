import os
import numpy as np
import cv2

# 加载和处理分类的数据


def load_data(path):
    filenames = os.listdir(path)
    samples = np.empty((0, 900))
    labels = []
    for filename in filenames:
        dir_path = path + '/' + filename
        img_paths = os.listdir(dir_path)
        for img_path in img_paths:
            labels.append(filename)
            img = cv2.imread(dir_path + '/' + img_path, 0)
            sample = img.reshape((1, 900)).astype(np.float32)
            samples = np.append(samples, sample, 0)
    # 把样本像素矩阵转化为二位举证
    samples = samples.astype(np.float32)
    unique_labels = list(set(labels))
    unique_ids = list(range(0, len(unique_labels)))
    label_id_map = dict(zip(unique_labels, unique_ids))
    id_label_map = dict(zip(unique_ids, unique_labels))
    # 把label 转化为 id
    label_ids = list(map(lambda x: label_id_map[x], labels))
    # print(len(labels), len(label_ids))
    # 把id 转化为二位矩阵
    label_ids = np.array(label_ids).reshape((-1, 1)).astype(np.float32)
    # sample、label_ids是模型训练的使用的数据结构
    return [samples, label_ids, id_label_map]


# 分割识别的二维码为单个字符


def split_img(path):
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

    boxs = []
    for i in range(1, len(index_list)):
        img2 = img1[:, index_list[i - 1]:index_list[i]]
        res = cv2.resize(img2, (30, 30), interpolation=cv2.INTER_CUBIC)
        boxs.append(res)
    return boxs


def get_code(path):
    # 获取处理的数据
    [samples, label_ids, id_label_map] = load_data('img')
    model = cv2.ml.KNearest_create()
    # print(len(samples), len(label_ids))
    # 用数据训练模型
    model.train(samples, cv2.ml.ROW_SAMPLE, label_ids)
    boxs = split_img(path)
    result = []
    # 用训练的模型识别分割好的验证码
    for box in boxs:
        sample = box.reshape((1, 900)).astype(np.float32)
        ret, results, neighbours, distances = model.findNearest(sample, k=3)
        label_id = int(results[0, 0])
        label = id_label_map[label_id]
        result.append(label)

    return result


print(get_code('19.jpg'))
