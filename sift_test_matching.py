import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

#img = cv2.imread('home.jpg')
img1 = cv2.imread('/media/sf_flickr-images/im20014.jpg')
gray1= cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('/media/sf_flickr-images/im20015.jpg')
gray2= cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# Initiate SIFT detector
#sift = cv2.SIFT()
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(gray1,None)
kp2, des2 = sift.detectAndCompute(gray2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)

plt.imshow(img3),plt.show()


#dictionarySize = 4
#BOW = cv2.BOWKMeansTrainer(dictionarySize)

#img2=cv2.drawKeypoints(gray,kp,img)
#img2=cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

#plt.imshow(img2)
#plt.show()

#v2.imwrite('sift_keypoints.jpg',img2)

#cv2.imwrite('sift_keypoints.jpg',des)
#with open('des.pickle', 'wb') as f:
#    pickle.dump(des, f)

#BOW.add(des)
#dictionary created
#dictionary = BOW.cluster()
#print(dictionary)
