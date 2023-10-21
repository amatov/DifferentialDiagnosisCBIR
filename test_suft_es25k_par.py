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
import itertools
import glob
import time
import pprint
from math import ceil
from tqdm import tqdm
from numpy import average, median
import multiprocessing as mp

index_name = 'idx-sift'

# To specify num clusters, use this command:
# NUM_CLUSTERS=200 python ./test_sift_es.py
num_clusters = int(os.environ.get('NUM_CLUSTERS', 5000))
model_path = './out/model.sift-%s.pickle' % num_clusters

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-l', '--limit',
            help = 'how many images to analyze (-1 means no limit)',
            default = -1,
            type = int,
    )
    parser.add_argument(
            '-p', '--path',
            help = 'path to images directory',
            default = './flickr-images',
            type = str,
    )
    cpu_count = mp.cpu_count()
    parser.add_argument(
            '--cpus',
            help = 'number of cores to use (defaults to using all cores: %d)' % cpu_count,
            default = cpu_count,
            type = int,
    )
    args = parser.parse_args()
    if args.limit < 0:
        args.limit = None

    start = time.time()
    data = []
    try:
        analyze(
                basepath = args.path,
                limit = args.limit,
                cpus = args.cpus,
                progress_fn = create_progress_fn(),
                results = data,
        )
    except KeyboardInterrupt:
        # If analysis interrupted by Ctrl-C continue with files analysed so far
        pass
    print('%s images analyzed in ~%.0fs' % (len(data), time.time() - start))
    if len(data) == 0:
        exit()
    print('')

def create_progress_fn():
    memo = {
        'pbar': None,
        'i': 0,
    }

def index_source_image(es, file_id, terms):
    # by default we connect to localhost:9200
    es.indices.create(index=index_name, ignore=400)
    es.index(index=index_name, doc_type="artwork", id=file_id, body={"tags": list(set(terms))})

def build_kmeans_model(descriptors, num_clusters=10000):
    print('starting clustering (%s clusters)' % num_clusters)
    kmeans = skcluster.KMeans(num_clusters)
    start = time()
    idxs = kmeans.fit_predict(descriptors)
    print('done in %0.3fs' % (time() - start))
    return idxs, kmeans

path = './flickr-images/*.jpg'
files=glob.glob(path)

sift = cv2.xfeatures2d.SIFT_create()

def buildDictionary():
    dictionarySize = num_clusters
    print('building dictionary of %s visual words' % dictionarySize)
    #BOW = cv2.BOWKMeansTrainer(dictionarySize)
    descriptors=[]

    for filename in itertools.islice(files_iter, 0, limit):
        filenames.append(filename)

    for filename in files:
        image = cv2.imread(filename,0)
        print('processing/features %s...' % filename,)
        #plt.imshow(image), plt.show()
        #gray = cv2.cvtColor(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        kp, des= sift.detectAndCompute(image, None)
        descriptors.extend(des)
        #BOW.add(des)
    indexes, model = build_kmeans_model(descriptors, dictionarySize)
    print(len(indexes))
    print(len(descriptors))
    print(model.cluster_centers_)
    with open(model_path, 'wb+') as f:
        pickle.dump(model, f)
    return model

#dictionary created
#dictionary = BOW.cluster()
try:
    with open(model_path, 'rb') as f:
        model = pickle.load( f)
except FileNotFoundError:
    model = buildDictionary()

es = Elasticsearch()

# REPLACE FILENAME with a larger library to index over 25k
#path = './1000testImages/*.jpg'
path = './images/moreImages/*.jpg'
files=glob.glob(path)

for filename in files:
    image = cv2.imread(filename,0)
    print('processing/indexing %s...' % filename,)
    #plt.imshow(image), plt.show()
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp, descs= sift.detectAndCompute(image, None)
    words = [int(i) for i in model.predict(descs)]
    index_source_image(es,filename,words)
    print(set(words))
    #img2=cv2.drawKeypoints(image,kp,image,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #plt.imshow(img2)
    #plt.show()
    print('len kp %s...' % len(kp),)
