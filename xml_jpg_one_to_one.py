"""
description：
假设文件夹file_dir中包含我们需要的png和jpg图片以及对应的xml文件，同时，充斥着一些无效文件（eg：jpg或png无对应的xml文件，xml无对应的jpg源图片，其他格式如.txt等文件），
该脚本做的工作是：1.将无效文件移至file_dir_useless文件夹，有效文件位置不动
               """
import os
import numpy as np
from PIL import Image
import xml.dom.minidom
import sys
import shutil
file_dir = "D:/ship/ship_name/test_oldversion/all_picture"
file_dir_useless = "D:/ship/ship_name/test_oldversion/useless"
def xml_jpg_one_to_one(file_dir):
    list_jpg = []
    list_xml = []
    for root, dirs, files in os.walk(file_dir):

            """循环当前目录下的文件名，将jpg和png图片名循环存入list_jpg列表中，xml文件名循环存入list_xml列表中"""
            for file in files:
                if os.path.splitext(file)[1] == '.jpg' or os.path.splitext(file)[1] == '.png':
                    list_jpg.append(file)
                if os.path.splitext(file)[1] == '.xml':
                    list_xml.append(file)

            """循环当前目录下的文件名，若当前文件名（不含格式）不在集合（图片名和xml文件名交集（不含格式）组成的集合）中，则将该文件移至新的文件夹file_dir_useless里"""
            for efile in files:
                if os.path.splitext(efile)[0] in list(set([i[:-4] for i in list_jpg]) & set([j[:-4] for j in list_xml])):
                    continue
                else:
                    #os.remove(root+"/"+efile)
                    shutil.move(root+"/"+efile, file_dir_useless)

myfile = xml_jpg_one_to_one(file_dir)