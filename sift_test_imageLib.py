import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import os
import glob
import sys
import getHistogram
#import sklearn.cluster as skclusters
#from sklearn.decomposition.pca import PCA
#from elasticsearch import Elasticsearch

path = './flickr-images/*.jpg'
files=glob.glob(path)

sift = cv2.xfeatures2d.SIFT_create()

dictionarySize = 5#30
BOW = cv2.BOWKMeansTrainer(dictionarySize)

for filename in files:
    image = cv2.imread(filename,0)
    print('processing %s...' % filename,)
    #plt.imshow(image), plt.show()
    #gray = cv2.cvtColor(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    kp, des= sift.detectAndCompute(image, None)
    BOW.add(des)
    #plt.figure() # <- makes a new figure and sets it active 
    #plt.hist(des) # <- finds the current active axes/figure and plots to it
    #plt.show()
    #plt.title('Codeword histogram')
    #plt.xlabel(filename)
    #plt.ylabel('Five bins/codewords/k-means')
    #plt.axis([0, 1000, 0, 10]) #occurences, bins
    #plt.savefig('hist') # <- saves the currently active figure 

#dictionary created
dictionary = BOW.cluster()

for filename in files:
    image = cv2.imread(filename,0)
    print('processing %s...' % filename,)
    #plt.imshow(image), plt.show()
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp, des= sift.detectAndCompute(image, None)
    img2=cv2.drawKeypoints(image,kp,image,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img2)
    plt.show()
    print('len kp %s...' % len(kp),)
    histo = getHistogram.getHistogram(dictionary, des)
    print(histo)

    y = histo
    N = len(y)
    x = range(N)
    width = 1/1.5
    plt.bar(x, y, width, color="blue")
    #plt.hist(histo) # <- finds the current active axes/figure and plots to it
    plt.show()

img = cv2.imread('./flickr-images/im20014.jpg')


with open('des.test.pickle', 'wb') as f:
    pickle.dump(des, f)

#print(dictionary)

#dims = dictionary
#height = dims[0]
#width = dims[1]
#print('vocabulary size %s...' %sys.getsizeof(dictionary))
