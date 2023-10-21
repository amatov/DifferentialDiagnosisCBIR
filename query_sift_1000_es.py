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

path = './1000testImages/*.jpg'
files=glob.glob(path)

sift = cv2.xfeatures2d.SIFT_create()

#model = pickle.load('model.pickle')
with open('./out/model.sift-5000.pickle', 'rb') as f:
    model = pickle.load( f)

es = Elasticsearch()
start = time()

filename = './1000testImages/surfer_distortion_3_a.jpg'
#filename = './mirflickr/1000images/im20014.jpg'

image = cv2.imread(filename,0)
print('processing query image %s...' % filename,)
kp, descs= sift.detectAndCompute(image, None)
words = [int(i) for i in model.predict(descs)]
results = query_source_images(es, words)

print("matching analysis done in %0.3fs" % (time() - start))
#print(results)

#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1)
#imgplot = plt.imshow(image)

# displaying the input query image
s(filename)

for result in results['hits']['hits']:
    #pp.pprint(result)
    #hits = result['hits']
    id = result['_id']
    score = result['_score']
    if score > 1:
        print( id, score)
        # displaying the matchesß
        s(id)
