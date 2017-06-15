import numpy as np
import cv2
from scipy.misc import toimage

#Standard Skin Detection using RGB,HSV and YCrCB space.	
def findValues(img):
	#Split the BGRA image into its Components
	b_values,g_values,r_values,a_values = cv2.split(img)
	#Convert BGRA image to BGR image
	img1 = cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
	#Comvert the BGR image to HSV
	hsv_img = cv2.cvtColor(img1,cv2.COLOR_BGR2HSV)
	#Split the HSV image into components
	h_values,s_values,v_values = cv2.split(hsv_img)
	#Convert the BGR image to YCRVb
	ycrcb_img = cv2.cvtColor(img1,cv2.COLOR_BGR2YCrCb)
	#Split the YCrCb image into components
	y_values,cr_values,cb_values = cv2.split(ycrcb_img)
	
	rows,columns = img.shape[0],img.shape[1]
	
	#Calculate the thresholdsaccording to the conditions
	h1 = h_values<=50
	h2 = h_values>=0
	h = np.logical_or(h1,h2)
	s1 = s_values<=0.68
	s2 = s_values>=0.23
	s = np.logical_or(s1,s2)
	r = r_values>95
	g = g_values>40
	b = b_values>20
	rg = r_values>b_values
	rb = r_values>b_values
	r_g = np.absolute(r_values-b_values)>15
	a = a_values>15
	cr = cr_values>135
	cb = cb_values>85
	y = y_values>80
	cr1 = (cr_values<=(1.5862*cb_values)+20)
	cr2 = (cr_values>=(0.3448*cb_values)+76.2069)
	cr3 = (cr_values >= (-4.5652*cb_values)+234.5652)
	cr4 = (cr_values <= (-1.15*cb_values)+301.75)
	cr5 = (cr_values <= (-2.2857*cb_values)+432.85)	
	
	#Two conditions to make
	cond1 = [r,g,b,rg,rb,r_g,a]
	cond2 = [r,g,b,rg,rb,r_g,a,cr,cb,y,cr1,cr2,cr3,cr4,cr5]
	
	temp1 = np.logical_and(h,s)
	for i in cond1:
		temp1 = np.logical_and(temp1,i)
	
	temp2 = np.logical_and(h,s)
	for i in cond2:
		temp2 = np.logical_and(temp2,i)
	
	#Join the two conditions
	final = np.logical_or(temp1,temp2)
	for i in range(final.shape[0]):
		for j in range(final.shape[1]):
			#Mask the pixel if it does not meet the conditions
			if final[i][j] == False:
				img[i][j] = (0,0,0,255)
	return img
	#toimage(img).show()

if __name__ == "__main__":
	
	img = cv2.imread("imag3.png",-1)
	skinImg = findValues(img)
	toimage(skinImg).show()
