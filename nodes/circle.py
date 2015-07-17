#!/usr/bin/env python

import rospy
import roslib
from sensor_msgs.msg import Image

import os
import cv2
import numpy
from cv_bridge import CvBridge, CvBridgeError


class face_detector:

    green = 0
    yellow = 1
    red = 2

    def __init__(self):
        
        self.bridge = CvBridge()

        self.image_id = 0

        self.output_directory = '/run/shm/face/'
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        self.image_pub = rospy.Publisher("output",Image)
        rospy.init_node('face_detector')
        rospy.Subscriber('image', Image, self.process, queue_size=1)
        rospy.spin()

    def process(self, data):
        try:
            raw_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError, e:
            print e

        gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 1.5, 300, minRadius=50, maxRadius=200)
 		
        
        if circles is not None:
            circles = numpy.round(circles[0, :]).astype("int")
 
            for (x, y, r) in circles:
                #cv2.circle(raw_image, (x, y), r, (0, 255, 0), 4)
                #cv2.rectangle(raw_image, (x - r, y - r), (x + r, y + r), (0, 0, 255), 3)
                cut_im = raw_image[y-r:y+r, x-r:x+r]
                hsv_im = cv2.cvtColor(cut_im, cv2.COLOR_BGR2HSV)
                red = self.count_color(hsv_im, self.red)
                green = self.count_color(hsv_im, self.green)
                yellow = self.count_color(hsv_im, self.yellow)
                print red, green, yellow
        #try:
        #    self.image_pub.publish(self.bridge.cv2_to_imgmsg(cut_im, "bgr8"))
        #except CvBridgeError, e:
        #    print e

    def count_color(self, image, color):
        h,w,d = image.shape
        lower,upper = self.get_bound_color(color)

        mask = cv2.inRange(image, lower, upper)
        count_pixel = mask.reshape(h*w).tolist().count(255)

        return count_pixel*100.0/(h*w)

    def get_bound_color(self, color):
        if color == self.green:
            return  (numpy.array([65,50,0], dtype=numpy.uint8),
                    numpy.array([110,255,200], dtype=numpy.uint8))
        elif color == self.yellow:
            return  (numpy.array([20,50,0], dtype=numpy.uint8),
                    numpy.array([65,255,200], dtype=numpy.uint8))
        elif color == self.red:
            return  (numpy.array([0,50,0], dtype=numpy.uint8),
                    numpy.array([20,255,200], dtype=numpy.uint8))

if __name__ == '__main__':
    try:
        face_detector()
    except rospy.ROSInterruptException:
        pass
