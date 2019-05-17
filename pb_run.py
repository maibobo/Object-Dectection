#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from matplotlib import pyplot as plt
from IoU_calculate import compute_iou, original_box


class TOD(object):
    def __init__(self):
        self.PATH_TO_CKPT =r'D:\ship\ship_name\test\model\s_yuntai_resnet_graph.pb\frozen_inference_graph.pb'

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

    # Helper code
    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    def detect(self, image):
        with self.detection_graph.as_default():
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
            # config = tf.ConfigProto()
            # config.gpu_options.allow_growth = True
            # with tf.Session(graph=self.detection_graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            with tf.Session(graph=self.detection_graph) as sess:
                '''the array based representation of the image will be used later in order to prepare the
                   result image with boxes and labels on it.'''
                # (im_width, im_height) = image.size
                # image_np = np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
                '''Expand dimensions since the model expects images to have shape: [1, None, None, 3]'''
                image_np_expanded = np.expand_dims(image, axis=0)
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                '''Each box represents a part of the image where a particular object was detected.'''
                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                '''Each score represent how level of confidence for each of the objects.
                   Score is shown on the result image, together with the class label.'''
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                '''Visualization of the results of a detection.'''
                image=vis_util.visualize_boxes_and_labels_on_image_array(
                #vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    self.category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                # plt.figure(figsize=(1920,1080))
                # plt.imshow(image)

                print(compute_iou(original_box(), boxes))
                cv2.imwrite('D:/ship/ship_name/test/result_shipname/{}'.format(str(i)), image)
                # cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)#根据原图大小进行展示
                # #cv2.namedWindow("detection", cv2.WINDOW_NORMAL)#图片窗口可调节大小
                # cv2.imshow("detection", image)
                # cv2.waitKey(10)

if __name__ == '__main__':

    img_path = 'D:/ship/ship_name/test/boat/boat_eval'
  #  img_path = 'E:\label_backup\small_yuntai2017'
    for dirpath, dirnames, filenames in os.walk(img_path):

        for i in filenames:
            if i.endswith('.jpg'):
                path = os.path.join(img_path, i)
                #print(i)
                image = cv2.imread(path)
                detector = TOD()
                detector.detect(image)
            #print (os.path.join(dirpath, i))


#pyinstaller -F pb_run.py --add-data "C:/ProgramData/Anaconda3/envs/tensorflow_cpu/Lib/site-packages/tensorflow/python/tensorflow.python.__python._pywrap_tensorflow_internal.pyd";.




