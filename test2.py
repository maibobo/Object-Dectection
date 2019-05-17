# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os,shutil
'''求模型改进训练前后测试结果图片集的交集数量与非交集部分的图片'''

file_dir = "D:\\ship\\ship_name\\test\\gao_data\\test_result3_5_imgmix\\result_img";  '''注： 3.5带jpg，4以后不带'''
file_dir1 = "D:\\ship\\ship_name\\test\\gao_data\\test_result4_1_imgmix\\result_img"  # unrecognized_picture
J = []  # jpg_list
for root, dirs, files in os.walk(file_dir):
    """循环当前目录下的文件名，将jpg图片名循环存入J列表中"""
    for file in files:
        #if os.path.splitext(file)[1] == '.jpg':   ;'''注释掉因为全是jpg'''
            (filename, extension) = os.path.splitext(file)
            J.append(filename)
            #J.append(file)
J1 = []
for root, dirs, files in os.walk(file_dir1):
    """循环当前目录下的文件名，将jpg图片名循环存入J列表中"""
    for file in files:
       # if os.path.splitext(file)[1] == '.jpg':
            J1.append(file)
# print(J)
# print(J1)
list1 = list( set(J).intersection(set(J1)) )
print(len(list1))
a = list(set(J) - set(list1))
b = list(set(J1) - set(list1))

print( a )
print(len(a))
print( b )
print(len(b))

