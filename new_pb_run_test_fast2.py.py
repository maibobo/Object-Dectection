#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

INPUT_TXT_PATH=[
#"Path/haishi2017.txt",
#"Path/gaogang201704.txt",
"Path/gaogang201710.txt",
"Path/gaogang201709.txt",
"Path/gaogang201708.txt",
"Path/gaogang201707.txt",
"Path/gaogang201706.txt",
"Path/gaogang201705.txt",
]

OUTPUT_CSV_PATH=[
"LocResult/CSV/haishi2017_result.csv",
"LocResult/CSV/gaogang201704_result.csv",
"LocResult/CSV/gaogang201710_result.csv",
"LocResult/CSV/gaogang201709_result.csv",
"LocResult/CSV/gaogang201708_result.csv",
"LocResult/CSV/gaogang201707_result.csv",
"LocResult/CSV/gaogang201706_result.csv",
"LocResult/CSV/gaogang201705_result.csv",
]

OUT_IMG_PATH=[
#"LocResult/IMG/haishi2017(run)",
#"LocResult/IMG/gaogang201704(run)",
"LocResult/IMG/gaogang201710(run)",
"LocResult/IMG/gaogang201709(run)",
"LocResult/IMG/gaogang201708(run)",
"LocResult/IMG/gaogang201707(run)",
"LocResult/IMG/gaogang201706(run)",
"LocResult/IMG/gaogang201705(run)",
]


MODEL_PATH=r"D:\ship\ship_name\test_oldversion\model\alldata_rcnn_inceptionv2_frozen_inference_graph_expand_x20.pb\frozen_inference_graph.pb"

LABELS_PATH=r"D:\ship\ship_name\test_oldversion\data\boat_label_map.pbtxt"

## 读取图像，解决imread不能读取中文路径的问题
def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    ##cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img

class TOD(object):
    def __init__(self):
        self.PATH_TO_CKPT =MODEL_PATH
        self.PATH_TO_LABELS = LABELS_PATH
        self.NUM_CLASSES = 1   #类型
        self.detection_graph = self._load_model()
        self.category_index = self._load_label_map()

    def _load_model(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph


    def _load_label_map(self):
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=self.NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index

    def detect(self, path,idx):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                out_csv = OUTPUT_CSV_PATH[idx]
                f = open(out_csv, "a+")
                f.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")

                out_img_dir=OUT_IMG_PATH[idx]
                if not os.path.exists(out_img_dir):
                    os.makedirs(out_img_dir)

                i=1
                for one_path in path:
                    one_path = one_path[:-1]
                    if (os.path.exists(one_path)):
                        if (os.path.getsize(one_path)):
                            image = cv_imread(one_path)
                            # image=np.array(image,dtype=int)
                            h = image.shape[0]
                            w = image.shape[1]
                            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                            image_np_expanded = np.expand_dims(image, axis=0)
                            image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                            boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                            scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                            classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                            num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                            # Actual detection.
                            (boxes, scores, classes, num_detections) = sess.run(
                                [boxes, scores, classes, num_detections],
                                feed_dict={image_tensor: image_np_expanded})
                            # Visualization of the results of a detection.
                            # image,class_name=vis_util.visualize_boxes_and_labels_on_image_array(

                            image, box, classes_name = vis_util.visualize_boxes_and_labels_on_image_array(
                                # vis_util.visualize_boxes_and_labels_on_image_array(
                                image,
                                np.squeeze(boxes),
                                np.squeeze(classes).astype(np.int32),
                                np.squeeze(scores),
                                self.category_index,
                                use_normalized_coordinates=True,
                                line_thickness=8)
                            if (str(classes_name) == str("shipname")):
                                for ele in box:
                                    print(ele)
                                    ymin, xmin, ymax, xmax = ele
                                    outline = '%s,%d,%d,%s,%d,%d,%d,%d\n' % (
                                    one_path, w, h, str(classes_name), int(xmin * w), int(ymin * h), int(xmax * w),
                                    int(ymax * h))
                                    f.write(outline)
                                img_name=out_img_dir+"/{}.jpg".format(str(i))
                                cv2.imencode('.jpg', image)[1].tofile(img_name)

                            i = i + 1
                            print("### Processing %d img:%s ###" % (i, one_path))
                        else:
                            print("Image is empty : %s !" % one_path)
                    else:
                        print("Path is not exists:%s ! " % one_path)


                    #cv2.imencode('.jpg', image)[1].tofile('F:/ShipCol_TF/ShipnameLoc_Img_Result_55103/{}.jpg'.format(str(i)))

                #else:
                    #cv2.imwrite('F:/xichuanzhu/test/processed_picture/unrecognized_boat/{}'.format(str(i)), image)
                    #cv2.imencode('.jpg', image)[1].tofile('D:/ship/mcol_loc/result_test/unrcg/{}.jpg'.format(str(i)))

                #print(class_name)

        #cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)#根据原图大小进行展示
                #cv2.namedWindow("detection", cv2.WINDOW_NORMAL)#图片窗口可调节大小
                #cv2.imshow("detection", image)
                #cv2.waitKey(10)

if __name__ == '__main__':

    for index, path_txt in enumerate(INPUT_TXT_PATH):
        f=open(path_txt,'r')
        path_list=f.readlines()

        detecotr = TOD()
        detecotr.detect(path_list, index)

    # for dirpath, dirnames, filenames in os.walk(img_path):
    #     for i in filenames:
    #         if i.endswith('.jpg'):
    #             if i.endswith('.jpg'):
    #                 path = os.path.join(img_path, i)
    #                 #print(i)
    #                 #image = cv2.imread(path)
    #
    #                 detecotr = TOD()
    #                 detecotr.detect(path)
    #         #print (os.path.join(dirpath, i))