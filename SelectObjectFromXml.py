# -*- coding: utf-8 -*-
"""
Created on Thu Dec 6 16:30 2018
@author: 林盛梅
description：
对源文件目录sourceDir中的xml进行修改，主要是保留“船名”部分，删除其余object
将中文标签名和文件夹名改成指定名称
将修改后的xml保存在targetDir中
"""
import xml.etree.cElementTree as ET
import os
import pandas as pd
import numpy as np
from PIL import Image
import shutil

sourceDir = 'xml_test1' #源文件目录
targetDir = 'xml_test_result'
folderName = 'VOC2019'
classNameCh = '船名'
classNameEn = 'shipname'

"""
定义一个copyFiles函数，将sourceDir路径下的所有文件复制到targetDir文件夹下 
"""
def copyFiles(sourceDir):
    if not os.path.exists(targetDir):
        os.makedirs(targetDir)
    for files in os.listdir(sourceDir):
        xmlfile = sourceDir + "/" + files
        if os.path.splitext(xmlfile)[1] == '.xml':
            shutil.copy(xmlfile,targetDir)

def SelectObjectFromXml(file_dir):
    for root1, dirs, files in os.walk(file_dir):
        for mfile in files:
            if os.path.splitext(mfile)[1] == '.xml':
                xmlfile = root1 + "/" + mfile
                tree = ET.parse(xmlfile)
                root = tree.getroot()  # 获得root节点
                #data为字典，xmin代表键，item代表内容
                data = dict([(item[4][0].text, item) for item in root.findall('object')])

                #根据键对字典进行从小到大排序，注意，此时的键是数字字符，是str类型，而不是int类型
                # 比较的是int(str)的大小，str比较大小是比较str[0]的大小
                d = sorted(data.items(), key=lambda k: int(k[0]))

                #排序后的d变成了顺序列表
                for i in d:
                    #删除当前的object，并将其append到最后一个object后面，循环过后，便成为了顺序object（按照xmin排序）
                    root.remove(i[1])
                    root.append(i[1])
                tree.write(root1 + '/' + mfile)

                # 从后向前删除汉字object，若从前向后删除，删除root[i]后，root[i+1]将变成root[i]，不是很好循环。
                i = len(root) - 1
                while (i > 5):
                    str1 = root[i][0].text  # 坐标数字
                    if (str1!=classNameCh):
                        # print(root[i][0].text)#打印的应该都是汉字（非数字）
                        del root[i]
                        i = i - 1
                    else:  # else不能忘记，不然进入死循环
                        i = i - 1


                tree.write(root1 + '/' + mfile, "UTF-8")

def ModifyCh2En(xml_dir):
    for files in os.listdir(xml_dir):
        xmlfile = xml_dir + "/" + files
        if os.path.splitext(xmlfile)[1] == '.xml':
            tree = ET.parse(xmlfile)
            root = tree.getroot()  # 获得root节点

            sub1=root.find("folder")
            sub1.text=folderName

            for sub2 in root.findall("object"):
                subsub2 = sub2.find('name')
                subsub2.text = classNameEn

            if root.find("object"):
                tree.write(xmlfile)
            else:
                os.remove(xmlfile)


copyFiles(sourceDir)
SelectObjectFromXml(targetDir)
ModifyCh2En(targetDir)


