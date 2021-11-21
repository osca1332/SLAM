import time
import cv2
from display import Display
import numpy as np


W = 1920//2
H = 1080//2

disp = Display(W,H)


class FeatureExtractor(object):

	def __init__(self):
		self.orb = cv2.ORB_create(100)

	def extract(self, img):
		feats= cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)
		
		kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats]
		des = self.orb.compute(img, kps)
		
		return kps, des


fe = FeatureExtractor()

def process_frame(img):
	img = cv2.resize(img, (W,H))

	kps, des = fe.extract(img)

	for p in kps:
		u,v = map(lambda x: int(round(x)), p.pt)
		cv2.circle(img, (u,v), color=(0,255,0), radius=1, thickness= -1)

	disp.paint(img)






if __name__ == "__main__":
	cap = cv2.VideoCapture("test.mp4")
	while cap.isOpened():
		ret, frame = cap.read()
		if ret == True:
			process_frame(frame)
		else:
			break