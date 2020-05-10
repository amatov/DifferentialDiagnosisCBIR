import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import os
import glob
import sys
import getHistogram
import sklearn.cluster as skcluster
from sklearn.decomposition.pca import PCA
from elasticsearch import Elasticsearch
from time import time
index_name = 'idx-surf'
def index_source_image(es, file_id, terms):
    # by default we connect to localhost:9200
    es.indices.create(index=index_name, ignore=400)
    es.index(index=index_name, doc_type="artwork", id=file_id, body={"tags": list(set(terms))})

def build_kmeans_model(descriptors, num_clusters=10000):
    print("starting clustering")
    kmeans = skcluster.KMeans(num_clusters, max_iter=10000)
    start = time()
    idxs = kmeans.fit_predict(descriptors)
    print("done in %0.3fs" % (time() - start))
    return idxs, kmeans

path = '/media/sf_flickr-images/*.jpg'
files=glob.glob(path)

surf = cv2.xfeatures2d.SURF_create(1000)

def buildDictionary():
    dictionarySize = 5000#30
    #BOW = cv2.BOWKMeansTrainer(dictionarySize)
    descriptors=[]
    for filename in files:
        image = cv2.imread(filename,0)
        print('processing %s...' % filename,)
        #plt.imshow(image), plt.show()
        #gray = cv2.cvtColor(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        kp, des= surf.detectAndCompute(image, None)
        descriptors.extend(des)
        #BOW.add(des)
    indexes, model = build_kmeans_model(descriptors, dictionarySize)
    print(len(indexes))
    print(len(descriptors))
    print(model.cluster_centers_)
    with open('model.surf.pickle', 'wb') as f:
        pickle.dump(model, f)
    return model

#dictionary created
#dictionary = BOW.cluster()
try:
    with open('model.surf.pickle', 'rb') as f:
        model = pickle.load( f)
except:
    model = buildDictionary()

es = Elasticsearch()

for filename in files:
    image = cv2.imread(filename,0)
    print('processing %s...' % filename,)
    #plt.imshow(image), plt.show()
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp, descs= surf.detectAndCompute(image, None)
    words = [int(i) for i in model.predict(descs)]
    index_source_image(es,filename,words)
    print(set(words))
    #img2=cv2.drawKeypoints(image,kp,image,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #plt.imshow(img2)
    #plt.show()
    print('len kp %s...' % len(kp),)
