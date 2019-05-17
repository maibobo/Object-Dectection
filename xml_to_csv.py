
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_csv(path):

    xml_list = []

    print("*********",glob.glob(path + '/*.xml'))

    for xml_file in glob.glob(path + '/*.xml'):

        tree = ET.parse(xml_file)

        root = tree.getroot()

        for member in root.findall('object'):

            value = (root.find('filename').text,int(root.find('size')[0].text),#    //如果生成的xml项后面没有图片格式声明，记得这里加上

                     int(root.find('size')[1].text),

                     member[0].text,

                     int(member[4][0].text),

                     int(member[4][1].text),

                     int(member[4][2].text),

                     int(member[4][3].text)

                     )

            xml_list.append(value)

    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

    xml_df = pd.DataFrame(xml_list, columns=column_name)

    return xml_df

# def main():
#     #image_path = os.path.join(os.getcwd(), 'annotations')
#
#     image_path1 = r'G:/labelImg/API_config/models1/test/boat_xml/eval'    #改这里的xml路径
#     image_path2 = r'G:/labelImg/API_config/models1/test/boat_xml/eval'
#     xml_df = xml_to_csv(image_path)
#     xml_df.to_csv('boat_eval.csv', index=None)    #生成的csv文件
#
#     print('Successfully converted xml to csv.')


def main():
    #for directory in ['train','eval']:
        xml_path = r'D:\ship\ship_name\test\gao_data\test_img1705\test_xml'
        xml_df = xml_to_csv(xml_path)
        xml_df.to_csv('D:/ship/ship_name/test/gao_data/Gao1705_resnet/boat_name_test.csv', index=None)
        print('Successfully converted xml to csv.')


main()

