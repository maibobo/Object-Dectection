"""
description：
假设文件夹test1中包含我们需要的png和jpg图片以及对应的xml文件，同时，充斥着一些无效文件（eg：jpg或png无对应的xml文件，xml无对应的jpg源图片，其他格式如.txt等文件），
该脚本做的工作是：1.将无效文件移至test3文件夹，有效文件位置不动
               2.通过有效的xml文件，将对应图片中的文字字符（中文汉字、数字、英文字母）等切割出来，分别保存在以其文字字符命名的文件夹中，并将这些文件夹保存在test2里
"""
import os
import numpy as np
from PIL import Image
import xml.dom.minidom
import sys
import shutil

def xml_jpg_one2one(file_dir):

    J=[]
    X=[]
    for root, dirs, files in os.walk(file_dir):

            """循环当前目录下的文件名，将jpg和png图片名循环存入J列表中，xml文件名循环存入X列表中"""
            for file in files:
                if os.path.splitext(file)[1] == '.jpg' or os.path.splitext(file)[1] == '.png':
                    J.append(file)
                if os.path.splitext(file)[1] == '.xml':
                    X.append(file)

            """循环当前目录下的文件名，若当前文件名（不含格式）不在集合（图片名和xml文件名交集（不含格式）组成的集合）中，则将该文件移至新的文件夹里"""
            for efile in files:
                if os.path.splitext(efile)[0] in list(set([X[:-4] for X in J]) & set([x[:-4] for x in X])):
                    continue
                else:
                    #os.remove(root+"/"+efile)
                    shutil.move(root+"/"+efile, "I:/船名标注/GaoGang201705rest")

myfile=xml_jpg_one2one("I:/船名标注/GaoGang201705")