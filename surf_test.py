import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

#img = cv2.imread('home.jpg')
img = cv2.imread('./flickr-images/im20014.jpg')

#plt.imshow(img)
#plt.show()

gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Create SURF object; can specify params here or later.
# set Hessian Threshold to a thousand to eliminate low certainty features
surf = cv2.xfeatures2d.SURF_create(1000)
#kp = surf.detect(gray,None)
kp, des = surf.detectAndCompute(gray,None)

img2=cv2.drawKeypoints(gray,kp,img)

dictionarySize = 5
BOW = cv2.BOWKMeansTrainer(dictionarySize)
plt.imshow(img2)
plt.show()

cv2.imwrite('surf_keypoints.jpg',img2)
with open('des.surf.pickle', 'wb') as f:
    pickle.dump(des, f)

BOW.add(des)
#dictionary created
dictionary = BOW.cluster()
print(dictionary)
