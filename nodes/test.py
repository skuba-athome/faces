import yannix

f = yannix.FacesDetect()
while(True):
	frame = f.GetFrame()
	f.Detect(frame)
