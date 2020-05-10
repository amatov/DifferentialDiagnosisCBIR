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

pp = pprint.PrettyPrinter(indent=4)
index_name = 'idx-surf'
def query_source_images(es, terms):
    return es.search(
        index=index_name, doc_type="artwork",
        body={"query": {"terms": {"tags": list(set(terms))}}})


path = '/media/sf_flickr-images/*.jpg'
files=glob.glob(path)

surf = cv2.xfeatures2d.SURF_create(1000)

#model = pickle.load('model.pickle')
with open('model.surf.pickle', 'rb') as f:
    model = pickle.load( f)

es = Elasticsearch()
start = time()

#filename = '/media/sf_flickr-images/im20014b.jpg'
#filename = '/media/sf_flickr-images/surfer_distortion_4_.jpg'
filename = '/media/sf_flickr-images/im20014.jpg'

image = cv2.imread(filename,0)
print('processing %s...' % filename,)
kp, descs= surf.detectAndCompute(image, None)
words = [int(i) for i in model.predict(descs)]
results = query_source_images(es, words)

print("done in %0.3fs" % (time() - start))
#print(results)

for result in results['hits']['hits']:
    #pp.pprint(result)
    #hits = result['hits']
    id = result['_id']
    score = result['_score']
    print( id, score)
