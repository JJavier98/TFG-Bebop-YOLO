from __future__ import print_function
import cv2 as cv2
from Queue import *
import threading
import time
import roslib
import imutils.video
try:
	roslib.load_manifest('TFG-Bebop-YOLO')
except:
	pass
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from videocaptureasync import VideoCaptureAsync
import os
current_path=os.path.dirname(os.path.abspath(__file__))

go_read=False

class VideoReader:

	def __init__(self, src, res='original', write_path=None, sync=False):
		self.path = src
		self.cam_path='/dev/video0'
		self.res = res
		self.cam_res=(426,240)
		self.frame=[]
		self.cam_frame=[]
		self.fps = 0.0
		self.ini_time = 0.0
		self.h = 0.0
		self.w = 0.0
		if self.path=='bebop':
			self.bridge = CvBridge()
			self.h = res[1]
			self.w = res[0]
		else:
			if sync:
				self.video_capture = cv2.VideoCapture(self.path)
				self.w = int(self.video_capture.get(3))
				self.h = int(self.video_capture.get(4))
			else:
				self.video_capture = VideoCaptureAsync(self.path)
				self.w = int(self.video_capture.cap.get(3))
				self.h = int(self.video_capture.cap.get(4))

		if res!='original':
			self.h = res[1]
			self.w = res[0]
			
		if write_path!=None:
			write_path=current_path+'/../../output/'+write_path+str(res)+'.avi'
			self.writable=True

			fourcc = cv2.VideoWriter_fourcc(*'XVID')
			self.out = cv2.VideoWriter(write_path, fourcc, 15, (self.w, self.h)) # path example: yolo_demo.avi
			#self.frame_index = -1
		else:
			self.writable=False

	def start(self):
		if self.path=='bebop':
			# start a thread to read frames from the file video video_capture
			self.t = threading.Thread(target=self.callback1, args=())
			self.t.daemon = True
			self.t.start()
			
			rospy.init_node('image_converter', anonymous=True)
		else:
			self.video_capture.start()
			
	def callback1(self):
		self.image_sub = rospy.Subscriber("/bebop/image_raw",Image,self.callback2)
	
	def callback2(self,data):
		if not go_read:
			try:
				img = self.bridge.imgmsg_to_cv2(data, "bgr8")
				self.frame = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
			except CvBridgeError as e:
				print(e)

	def read(self):
		go_read=True
		if self.path=='bebop':
			ret = True
		else:
			ret, self.frame = self.video_capture.read()
			if ret:
				self.frame = cv2.resize(self.frame, (self.w, self.h), interpolation=cv2.INTER_NEAREST)		

		value=self.frame
		go_read=False

		return (ret, value)

	def write(self, frame):
		if self.writable:
			try:
				self.out.write(frame)
			except:
				print('Bad output path or frame')
		else:
			pass

	def setIniTime(self):
		self.ini_time = time.time()

	def getFPS(self):
		self.fps = (self.fps + (1./(time.time()-self.ini_time))) / 2
		return self.fps

	def stopRead(self):
		try:
			self.video_capture.stop()
		except:
			pass

	def releaseWrite(self):
		try:
			self.out.release()
		except:
			pass