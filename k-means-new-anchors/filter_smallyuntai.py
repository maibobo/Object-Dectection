# -*- coding: utf-8 -*-

import os
#from sklearn.model_selection import train_test_split
import shutil

def filter_jpg(file_dir):

    J=[] #jpg_list
    J2=[]

    J1 = []
    J3 = []

    for root, dirs, files in os.walk(file_dir):
        """循环当前目录下的文件名，将jpg图片名循环存入J列表中"""
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                J.append(file)
                J2.append(os.path.splitext(file)[0])

            if os.path.splitext(file)[1] == '.xml':
                J1.append(file)
                J3.append(os.path.splitext(file)[0])



        # for jpg in J:
        #      if os.path.splitext(jpg)[0] in J3:
        #         shutil.copy(root + "/" + jpg, "/home/workstation/darknet/VOCdevkit/VOC2019/JPEGImages")

        for xml in J1:
            if os.path.splitext(xml)[0] in J2:
                shutil.copy(root + "/" + xml, "/home/workstation/darknet/VOCdevkit/VOC2019/Annotations")

def filter_jpg2(file_dir):

    J=[] #jpg_list
    J2=[]

    J1 = []
    J3 = []

    for root, dirs, files in os.walk(file_dir):
        """循环当前目录下的文件名，将jpg图片名循环存入J列表中"""
        for file in files:

            if os.path.splitext(file)[1] == '.xml':
                J1.append(file)
                J3.append(os.path.splitext(file)[0])

    for root, dirs, files in os.walk('/home/workstation/darknet/VOCdevkit/VOC2019/JPEGImages'):
        for file in files:
             if os.path.splitext(file)[0] not in J3:
                #shutil.copy(root + "/" + jpg, "/home/workstation/darknet/VOCdevkit/VOC2019/JPEGImages")
                os.remove(root + "/" + file)

        for xml in J1:
             if os.path.splitext(xml)[0] in J2:
               shutil.copy(root + "/" + xml, "/home/workstation/darknet/VOCdevkit/VOC2019/Annotations")

filter_jpg2("/home/workstation/darknet/VOCdevkit/VOC2019/Annotations")


