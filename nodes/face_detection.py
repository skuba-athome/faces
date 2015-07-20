#!/usr/bin/env python

import rospy
import roslib
from sensor_msgs.msg import Image
from std_msgs.msg import String

import os
import cv2
from cv_bridge import CvBridge, CvBridgeError


class face_detector:

    def __init__(self):
        
        self.bridge = CvBridge()

        face_cascade_file = rospy.get_param('~face_cascade_file', roslib.packages.get_pkg_dir('faces') + '/model/haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(face_cascade_file)
        self.image_id = 0
        print help(self.face_cascade)

        self.output_directory = '/run/shm/face/'
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

        self.image_pub = rospy.Publisher("output",Image)
        self.face_pub = rospy.Publisher("face_output",String)
        rospy.init_node('face_detector')
        rospy.Subscriber('image', Image, self.process, queue_size=1)
        rospy.spin()

    def process(self, data):
        try:
            raw_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError, e:
            print e

        small = cv2.resize(raw_image, (0,0), fx=1, fy=1)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        reject = [1, 3]
        we = [1.0, 1.0]
        #faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=2300)
        faces = self.face_cascade.detectMultiScale(gray, reject, we)
        for (x,y,w,h) in faces:
            cv2.rectangle(small,(x,y),(x+w,y+h),(255,0,0),2)
            print len(reject), len(we)

            self.image_id += 1
            face_img = small[y:y+h, x:x+w]
            img_name = self.output_directory + str(self.image_id)+'.jpg'
            print img_name
            cv2.imwrite(img_name, face_img)

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(small, "bgr8"))
        except CvBridgeError, e:
            print e

if __name__ == '__main__':
    try:
        face_detector()
    except rospy.ROSInterruptException:
        pass
