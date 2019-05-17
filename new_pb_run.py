#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

## 读取图像，解决imread不能读取中文路径的问题
def cv_imread(filePath):
    cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
    ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
    ##cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img

class TOD(object):
    def __init__(self):
        self.PATH_TO_CKPT =r'D:/ship/ship_name/test/model/gao1704_rcnn_inceptionv2_frozen_inference_graph_expand_x20.pb/frozen_inference_graph.pb'
        self.PATH_TO_LABELS = r'D:/ship/ship_name/test/data/boat_label_map.pbtxt'
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

    def detect(self, path):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                #f = open("Gao1704_resnet_result.csv", "a+")
                image = cv_imread(path)
                #image=np.array(image,dtype=int)
                #h = image.shape[0]
                #w = image.shape[1]
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
                #image,class_name=vis_util.visualize_boxes_and_labels_on_image_array(
                image,box,classes_name=vis_util.visualize_boxes_and_labels_on_image_array(
                #vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    self.category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                if (str(classes_name) == str("shipname")):
                    #for ele in box:
                        #ymin, xmin, ymax, xmax = ele
                        #outline='%s,%d,%d,%s,%d,%d,%d,%d\n'%(path,w,h,str(classes_name),int(xmin*w),int(ymin*h),int(xmax*w),int(ymax*h))
                        #f.write(outline)
                        cv2.imencode('.jpg', image)[1].tofile('D:/ship/ship_name/test/gao_data/test_result5_yangzh1/result_img/{}'.format(str(i)))
                else:
                #     #cv2.imwrite('F:/xichuanzhu/test/processed_picture/unrecognized_boat/{}'.format(str(i)), image)
                     cv2.imencode('.jpg', image)[1].tofile('D:/ship/ship_name/test/gao_data/test_result5_yangzh1/unrecognized_picture/{}'.format(str(i)));  '''可带中文,要去掉.jpg'''

                #print(class_name)

        #cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)#根据原图大小进行展示
                # cv2.namedWindow("detection", cv2.WINDOW_NORMAL)#图片窗口可调节大小
                # cv2.imshow("detection", image)
                # cv2.waitKey(10)

if __name__ == '__main__':

    #img_path = 'D:/ship/ship_name/test/gao_data/test_img1705/test_jpg'
    img_path = 'D:/ship/ship_name/test/gao_data/test_img_yangzh1'

    # f=open('D:/ship/ship_name/test/gao_data/test_img190125/test190125.txt','r')
    # data=f.readlines()
    # i=1
    #
    # for one_path in data:
    #     path=one_path[:-1]
    #     detecotr = TOD()
    #     detecotr.detect(path)
    #     i=i+1

    for dirpath, dirnames, filenames in os.walk(img_path):
        for i in filenames:
            if i.endswith('.jpg'):
                path = os.path.join(img_path, i)
                #print(i)
                #image = cv2.imread(path)

                detecotr = TOD()
                detecotr.detect(path)
            #print (os.path.join(dirpath, i))





