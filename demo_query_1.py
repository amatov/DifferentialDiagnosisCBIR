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
import pprint

def s(filepath):
    image = cv2.imread(filepath)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

pp = pprint.PrettyPrinter(indent=4)
index_name = 'idx-sift'
def query_source_images(es, terms):
    return es.search(
        index=index_name, doc_type="artwork",
        body={"query": {"terms": {"tags": list(set(terms))}}})

path = './flickr-images/*.jpg'
files=glob.glob(path)

sift = cv2.xfeatures2d.SIFT_create()

#model = pickle.load('model.pickle')
with open('out/model.sift-10000.pickle', 'rb') as f:
    model = pickle.load( f)

es = Elasticsearch()
start = time()

filename = './flickr-images/surfer_distortion_3_a.jpg'

image = cv2.imread(filename,0)
print('processing %s...' % filename,)
kp, descs= sift.detectAndCompute(image, None)
words = [int(i) for i in model.predict(descs)]
results = query_source_images(es, words)

print("done in %0.3fs" % (time() - start))
#print(results)

# Make figure
fig, (ax1, ax2,ax3) = plt.subplots(3, 1)
ax1.imshow(image)
ax1.set_title("query image")
ax2.imshow(image)
ax2.set_title("matched image")
ax3.imshow(image)
ax3.set_title("image ranked w next score")
#ax4.imshow(image)

#imgplot = plt.imshow(image)
#s(filename)

for result in results['hits']['hits']:
    #pp.pprint(result)
    #hits = result['hits']
    id = result['_id']
    score = result['_score']
    if score > 10:
        print( id, score)
        s(id)
