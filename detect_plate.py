"""
This python file contains a class doing
loading trained model, inference from these model ,
find max scores according to prediction of the class
and bounding boxes , normalize these bounding boxes
and return these normalized box coordinates
"""
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

from utils import label_map_util
from utils import visualization_utils as vis_util

class Detection():
    def __init__(self,IMAGE_NAME):
        sys.path.append("..")
        self.IMAGE_NAME = IMAGE_NAME
        self.MODEL_NAME = 'inference_graph'
        self.CWD_PATH = os.getcwd()
        self.PATH_TO_CKPT = os.path.join(self.CWD_PATH,self.MODEL_NAME,'frozen_inference_graph.pb')
        self.PATH_TO_LABELS = os.path.join(self.CWD_PATH,'training','labelmap.pbtxt')
        self.NUM_CLASSES = 1

        self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

    def run (self):
        """
        This function responsible of running below functions
        Returns:
            cropped = The image that probbally contains the predicted class (plate)
        """
        self.build_graph()
        self.get_tensors()
        self.capture()
        self.score , self.indice = self.get_max_score()
        self.bounding_boxes = self.get_bounding_boxes(self.indice)
        new_boxes = self.normalize_boxes(self.bounding_boxes)        
        cropped = self.crop_image(new_boxes)
        return cropped

        
    def build_graph(self):
        """
        This function builds detection graph by loading it into memory
        """
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            self.od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                self.serialized_graph = fid.read()
                self.od_graph_def.ParseFromString(self.serialized_graph)
                tf.import_graph_def(self.od_graph_def, name='')

        self.sess = tf.Session(graph=self.detection_graph)

    def get_tensors(self):
        """
        This function generates tensors 
        """
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

    def capture(self):
        """
        This function loads image that would read , also runs the tensorflow session to make a predict
        """
        self.build_graph()
        self.get_tensors()

        self.image = cv2.imread(self.IMAGE_NAME)
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image_rgb, axis=0) 
        
        (self.boxes, self.scores, self.classes) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes],
            feed_dict={self.image_tensor: image_expanded})
        
    def get_max_score(self):
        """
        After the prediction this fucntion calculates max score and its index in the prediction
        Returns :
            score: a float number
            max_index: an integer number
        """
        self.scores = np.squeeze(self.scores)
        max_indice = np.argmax(self.scores)
        return self.scores[max_indice],max_indice
    
    def get_bounding_boxes(self,max_indice):
        """
        This function returns bounding box according to max score's index
        Args:
            max_indice: an integer number about max_indice of predition
        Returns :
            bounding box: a numpy array  [0,4] 
        """
        return np.squeeze(self.boxes)[max_indice]

    def normalize_boxes(self,boxes):
        """
        This function makes bounding boxes normalized
        Args:
            boxes = bounding boxes
        Returns:
            new_boxes = normalized boxes
        """
        im_width = self.image.shape[0]
        im_height = self.image.shape[1]
         
        y_min = boxes[1]
        x_min = boxes[3]
        y_max = boxes[0]
        x_max = boxes[2]

        new_boxes = (x_min * im_width,x_max * im_width,y_min * im_height, y_max * im_height)
        return new_boxes
    
    def crop_image(self,new_boxes):
        """
        This function crops base_image according to predicted normalized bounding boxes
        Args:
            new_boxes: normalized bounding boxes
        Returns:
            corpped: cropped image
        """
        cropped = self.image[int(new_boxes[0]):int(new_boxes[1]), int(new_boxes[2]):int(new_boxes[3])]        
        return cropped
   