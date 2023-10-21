import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

#img = cv2.imread('home.jpg')
gray = cv2.imread('/media/sf_flickr-images/im20014.jpg',0)
img = cv2.imread('/media/sf_flickr-images/im20014.jpg')
#plt.imshow(img)
#plt.show()
#gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
#kp = sift.detect(gray,None)
kp, des = sift.detectAndCompute(gray,None)

dictionarySize = 5
BOW = cv2.BOWKMeansTrainer(dictionarySize)

#img2=cv2.drawKeypoints(gray,kp,img)
img2=cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.imshow(img2)
plt.show()

cv2.imwrite('sift_keypoints.jpg',img2)

#cv2.imwrite('sift_keypoints.jpg',des)
with open('des.sift.pickle', 'wb') as f:
    pickle.dump(des, f)

BOW.add(des)
#dictionary created
dictionary = BOW.cluster()
print(dictionary)
