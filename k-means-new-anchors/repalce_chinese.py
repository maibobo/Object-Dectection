# -*- coding:utf-8 -*-

# 将 系船柱 替换成 mcol

import os

xmldir = '/home/workstation/darknet/VOCdevkit/VOC2019/Annotations'
savedir = '/home/workstation/darknet/VOCdevkit/VOC2019/tmp'
xmllist = os.listdir(xmldir)
for xml in xmllist:
    if '.xml' in xml:
        fo = open(savedir + '/' + '{}'.format(xml), 'w')  #可加前缀'new_{}'
        print('{}'.format(xml))
        fi = open(xmldir + '/' + '{}'.format(xml), 'r')
        content = fi.readlines()
        for line in content:
            line = line.replace('系船柱','mcol')
            fo.write(line)
        fo.close()
        print('替换成功')

#如 mcol 为空字符串，就是删除