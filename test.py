# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os,shutil

# 1. 读取test.csv文件的第一列，并保存为list

# column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
# df=pd.read_csv('D:/ship/ship_name/test/gao_data/boat_name_test.csv')
# df=df['xmin', 'ymin', 'xmax', 'ymax']
# print('read_csv读取时指定索引：', df)

# list = df.values.tolist()
# image_id = []  # 存储的是要提取的数据
# for i in range(len(list)):
#     image_id.append(list[i][0])
# print(image_id)

df = pd.read_csv('D:/ship/ship_name/test/gao_data/boat_name_test.csv', \
                 usecols=['class', 'xmin', 'ymin', 'xmax', 'ymax'])
list1 = df.values.tolist()
# box = list(list1)  # 存储的是要提取的数据,一个列表
# for i in range(len(box)):
#     box.append(list[i][0])
print(list1)