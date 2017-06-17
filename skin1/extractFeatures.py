import numpy as np
import cv2
from scipy.misc import toimage
import skinDetector1

def normalizedSkinSize(img,encodeType):
	'''
	Find the number of normalized skin pixels in the image.
	Takes the number of skin pixels and divides by total pixels in the
	image.
	'''
	skinPixels = numSkinPixels(img,encodeType)
	height,width,channels = img.shape
	return skinPixels/(height*width)

def normalizedConnectedSkin(img,encodeType):
	
	'''
	Find the normalize skinpixel ratio for top 3 largest connected skin
	parts.
	Finds the top 3 contours first, gets the number of pixels in each
	then divides by total umber of skin pixels in the image.
	'''
	skinPixels = numSkinPixels(img,encodeType)
	
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	_,cnts,_ = cv2.findContours(gray.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(cnts,key = cv2.contourArea,reverse = True)[:3]
	cntSizes = [cnt.size for cnt in cnts]
	cntSizes = [cntSize/skinPixels for cntSize in cntSizes]
	cv2.drawContours(img,cnts,-1,(0,255,0),3)
	toimage(img).show()
	return cntSizes

def numSkinPixels(img,encodeType):
	if encodeType == 'png':
		threshold = (0,0,0,0)
	else:
		threshold = (0,0,0) 
	skinPixels = np.count_nonzero(img>threshold)
	return skinPixels

def tempRun(l):
	for img in l:
		filename = img
		nameSplit = filename.split('.')
		#Get the skin pixels in grey scale for the image
		skinImg = skinDetector1.runSkinDetection(filename)
		encodeType = skinDetector1.getEncodingType(filename)
		cntSize = normalizedConnectedSkin(skinImg,nameSplit[1])
		print("Sizes for file %s" %(filename))
		print(cntSize)
	
if __name__ == "__main__":
	#l = ['imag2.png']
	l = ['imag2.png','imag3.png','img1.jpeg','img1.png']
	tempRun(l)
