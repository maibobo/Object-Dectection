# -*- coding: utf-8 -*-

import os
from sklearn.model_selection import train_test_split
import shutil

def filter_jpg(file_dir):

    J=[] #jpg_list
    for root, dirs, files in os.walk(file_dir):
        """循环当前目录下的文件名，将jpg图片名循环存入J列表中"""
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                J.append(file)
        """分离训练集、验证+预测集 3:1:1"""
        result = train_test_split(J, test_size=0.4, random_state=0)
        train_list = result[0]
        retest_list = result[1]  #预测试图片列表

        """把标记出的训练xml文件复制到新文件夹即可，注意比较下复制和移动"""
        for train in train_list:
            shutil.copy(root + "/" + train, "D:/ship/ship_name/test/gao_data/test_img1707/train_jpg")

        """分离验证集、预测集"""
        result2 = train_test_split(retest_list, test_size=0.5, random_state=0)
        eval_list = result2[0]
        test_list = result2[1]

        """把标记出的eval、test图片复制到新文件夹即可，注意比较下复制和移动"""
        for eval in eval_list:
            shutil.copy(root + "/" + eval, "D:/ship/ship_name/test/gao_data/test_img1707/eval_jpg")
        for test in test_list:
            shutil.copy(root + "/" + test, "D:/ship/ship_name/test/gao_data/test_img1707/test_jpg")

filter_jpg("E:/label_backup/GaoGang201707")


