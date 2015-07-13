#!/usr/bin/env python

import rospy
import roslib
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge, CvBridgeError


class face_detector:

    def __init__(self):
        
        self.bridge = CvBridge()

        face_cascade_file = rospy.get_param('~face_cascade_file', roslib.packages.get_pkg_dir('faces') + '/model/haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(face_cascade_file)
        self.image_id = 0

        self.image_pub = rospy.Publisher("output",Image)
        rospy.init_node('face_detector')
        rospy.Subscriber('image', Image, self.process, queue_size=1)
        rospy.spin()

    def process(self, data):
        try:
            raw_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError, e:
            print e

        small = cv2.resize(raw_image, (0,0), fx=0.6, fy=0.6)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=2300)
        for (x,y,w,h) in faces:
            cv2.rectangle(small,(x,y),(x+w,y+h),(255,0,0),2)

            self.image_id += 1
            face_img = small[x:x+w, y:y+h]
            cv2.imwrite('~/face/' + str(self.image_id)+'.jpg', face_img)

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(small, "bgr8"))
        except CvBridgeError, e:
            print e

if __name__ == '__main__':
    try:
        face_detector()
    except rospy.ROSInterruptException:
        pass
