import cv

class FacesDetect:

	def __init__(self):
		self.storage = cv.CreateMemStorage(0)
		self.haarFace = cv.Load('haarcascade_frontalface_alt.xml')
		self.capture = cv.CaptureFromCAM(0)
		
	def Detect(self,image):
		faces = cv.HaarDetectObjects(image,self.haarFace,self.storage,1.2,2,0,(60,60))
		if faces:
			for face in faces :
				cv.Rectangle(image,(face[0][0],face[0][1]),(face[0][0]+face[0][2],face[0][1]+face[0][3]),cv.RGB(155,255,25),2)
		cv.NamedWindow("Face Detect",cv.CV_WINDOW_AUTOSIZE)
		cv.ShowImage("Face Detect",image)
		key = cv.WaitKey(10)
		if key == 27:
			exit()
	def GetFrame(self):
		return cv.QueryFrame(self.capture)


#if __name__=="__main__":
#	f = FacesDetect()
#	while(True):
#		frame = f.GetFrame()
#		f.Detect(frame)
		
