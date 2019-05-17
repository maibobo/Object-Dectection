# -*- coding: utf-8 -*-
import pandas as pd
from IoU_calculate import compute_iou, original_box
'''遍历原csv，和结果csv每一行，匹配相同shipname的行，计算box的交并比即iou，并保存到结果csv的最后一列，命名为新的标签列'''

test = pd.read_csv('D:/ship/ship_name/test/gao_data/test_img1704/boat_name_test.csv', \
                 usecols=['filename','xmin', 'ymin', 'xmax', 'ymax'])

result = pd.read_csv('D:/ship/ship_name/test/result_shipname/Gao1704_resnet/Gao1704_resnet_result.csv', \
                 usecols=['filename','xmin', 'ymin', 'xmax', 'ymax'])

print(test.drop_duplicates(subset=['filename']));   '''获取了删除重复行的结果，但并没有将结果保存为csv ？？？ '''

non_identical_test = test.drop_duplicates(subset=['filename'])
non_identical_result = result.drop_duplicates(subset=['filename'])
'''type( test、 non_identical_test ) —— <class 'pandas.core.frame.DataFrame'>'''
box1=[]
box2=[]
for index in non_identical_test.index:
    '''type( non_identical_test.loc[index].values[1:5] ) —— <class 'numpy.ndarray'>'''
    #box1[index] = non_identical_test.loc[index].values[1:5].tolist()
    box1.append([non_identical_test.loc[index].values[1:5].tolist()])
for index in non_identical_result.index:
    box2.append([non_identical_result.loc[index].values[1:5].tolist()])

'''不完全遍历'''
for index in non_identical_result.index:
    '''type( compute_iou(box1[index][0], box2[index][0]) ) —— <class 'numpy.float64'>'''
    '''type( box1[index]、 box1[index][0] ) —— <class 'list'>， 其中box1[index][0] 为[[767, 648, 997, 679]]'''
    print(box1[index][0], box2[index][0] )
    print( compute_iou(box1[index][0], box2[index][0]) )

# print(type(test))
# print(test.iloc[-1])

