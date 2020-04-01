"""
This python file contains a class doing
loading trained model, inference from these model ,
find max scores according to prediction of the class
and bounding boxes and calculate the plate strings
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import pandas as pd
from utils import label_map_util
from utils import visualization_utils as vis_util
from detect_plate import Detection


class Detection_Number_Plate:
    def __init__(self):
        sys.path.append("..")
        
        self.MODEL_NAME = 'inference_graph2'
        self.CWD_PATH = os.getcwd()
        self.PATH_TO_CKPT = os.path.join(self.CWD_PATH,self.MODEL_NAME,'frozen_inference_graph.pb')
        self.PATH_TO_LABELS = os.path.join(self.CWD_PATH,'training2','labelmap.pbtxt')
        
        self.NUM_CLASSES = 36

        self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)


    def run(self,image):
        """
        This function responsible of running below functions
        Args:
            image: an image that contain plate that would be predicted its numbers
        Returns:
            plate: the plate numbers
            score: the score about plate numbers
        """

        self.build_graph()
        self.get_tensors()
        self.predict(image)
        self.prediction = self.get_all_classes_prediction()
        prediction_max =self.find_max_prob(0.60)
        self.box_coordinates = self.find_bounding_box_coordinates(prediction_max)  
        plate, score = self.arrange_prediction()
        return plate,score

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

    def predict(self,image):        
        """
        This function reads image and predicts the classes ,scores and boxes
        Args: 
            image: an image that contains plate
        """
        self.build_graph()
        self.get_tensors()
              
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_expanded = np.expand_dims(image_rgb, axis=0) 
        
        (self.boxes, self.scores, self.classes) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes],
            feed_dict={self.image_tensor: image_expanded})
    
    def get_all_classes_prediction(self):
        """
        This functions calculates predictions of all classes and stack its to python dictionary
        Returns:
            all_prediction: a dictionary having all predictions of classes
        """
        classes_ = np.squeeze(self.classes)
        scores_ = np.squeeze(self.scores)

        max_classes = classes_ .shape[0]
        all_predictions = {}

        for i in range(max_classes):
         all_predictions[i] = {}

        for i in range(max_classes):
            max_indice = np.argmax(scores_)
            class_name = self.category_index[classes_[max_indice]]['name']
            all_predictions[i][class_name] = scores_[max_indice]
            scores_ = np.delete(scores_,max_indice)
            classes_ = np.delete(classes_,max_indice)
        return all_predictions
    
    def find_max_prob(self,threshold):
        """
        This function finds max probabilty of predicted classes according to the threshold
        Args:
            threshold: a float number
        Returns:
            new_dict: a dictionary that contains max probability of classes
        """
        copy = self.prediction.copy()
        new_dict = {}

        for i in range(len(copy)):
            new_dict[i] = {}
  
        for i,k  in enumerate(copy):
            for key in copy[k]:
             if copy[k][key] > threshold:      
              new_dict[i][str(key)] = copy[k][key]
             else:
                 pass
    
        return new_dict
    
    def find_bounding_box_coordinates(self,plate_dict):
        """
        This functions finds bounding box coordinates according to max probability of classes
        Args:
            plate_dict: a dictionary that contains max probability of classes 
        Returns:
            finding_boxes: a numpy array that contains bounding boxes according to max probablity of classes
        """
        boxes = np.squeeze(self.boxes)
        scores = np.squeeze(self.scores)
        plate_dict = plate_dict.copy()

        finding_boxes  = []

        for i,value in enumerate(scores):
            if value in plate_dict[i].values():
             index = np.where(scores == value)
             finding_boxes.append(boxes[index])
             
        return np.asarray(finding_boxes)
    
    def get_x_mins(self,box_coordinates):
        """
        This function gives xmin coordinates
        Args: 
            box_coordinates: a numpy array that contains bounding box
        Returns:
            x_mins: a numpy array that contains xmin coordinate of bounding box
        
        """
        box_coordinates = box_coordinates.copy()
        x_mins = []
        for b_list in box_coordinates:
            x_mins.append(b_list[0][1])

        return np.asarray(x_mins)
    
    def order_coordinates(self):
        """
        This function orders the bouinding box coordinates to acquire plate as ordered left-to-right
        Returns:
            sorted_array: a array of xmins coordinate of bounding boxes as sorted
            sorted_indices: indices that are sorted , numpy array
        """
        x_mins = self.get_x_mins(self.box_coordinates)
        sorted_indices = np.argsort(x_mins)
        sorted_array = x_mins[sorted_indices]

        return sorted_array, sorted_indices

    def arrange_prediction(self):
        """
        This function acquires the plate and score according to sorted predicted classes and scores
        Returns:
            plate: a string that contain plate as sorted
            score: a string that contain all classes scores bracket with "-"
        """
        sorted_coordinates , sorted_indices = self.order_coordinates()
        classes = np.squeeze(self.classes)
        scores = np.squeeze(self.scores)

        class_names = []
        scores_ = []

        for i in range(len(sorted_coordinates)):
         index = sorted_indices[i]
         class_index = classes[index]
         class_name = self.category_index[class_index]['name']
         class_names.append(class_name)
         scores_.append(scores[index])
        
        plate = ""
        score = ""
        for i in range(len(class_names)):
            plate += class_names[i]
            score += str(scores_[i])+"\n"

        return plate,score


