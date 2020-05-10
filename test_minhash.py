import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import os
import glob
import sys
import random

# ======== MinHash=======
# each image is represented by the set of words it contains.
# use the MinHash algorithm to calculate short
# signature vectors to represent the images. These MinHash signatures can
# then be compared quickly by counting the number of components in which the
# signatures agree. compare all possible pairs of images only for when there is a min number of matching sketches
#    - The MinHash algorithm is implemented using the random hash function
#      trick which prevents us from having to explicitly compute random
#      permutations of all of the words IDs.
# compare all MinHash signatures to one another.
#    - Compare MinHash signatures by counting the number of components in which
#      the signatures are equal only for pairs with a min number of mathcing sketches.
# Divide the number of matching components by the signature length to get a similarity value.
# Display pairs of documents / signatures with similarity greater than a threshold.

# number of components N in the resulting MinHash signatures = n * k where k is the number of sketches (5) and n is the sketch length (2)
numHashes = 10;
numSketches = 5;
lenSketches = numHashes/numSketches
# You can run this code for different portions of the dataset.
# It ships with data set sizes 100, 1000, 2500, and 10000.
numImages = 56 # 50 test images plus 6 distorted versions of SURFER

def build_kmeans_model(descriptors, num_clusters=10000):
    print("starting clustering")
    kmeans = skcluster.KMeans(num_clusters, max_iter=10000)
    start = time()
    idxs = kmeans.fit_predict(descriptors)
    print("done in %0.3fs" % (time() - start))
    return idxs, kmeans
path = '/media/sf_flickr-images/*.jpg'
files=glob.glob(path)

sift = cv2.xfeatures2d.SIFT_create()

def buildDictionary():
    dictionarySize = 5#30
    #BOW = cv2.BOWKMeansTrainer(dictionarySize)
    descriptors=[]
    for filename in files:
        image = cv2.imread(filename,0)
        print('processing %s...' % filename,)
        #plt.imshow(image), plt.show()
        #gray = cv2.cvtColor(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        kp, des= sift.detectAndCompute(image, None)
        descriptors.extend(des)
        #BOW.add(des)
    indexes, model = build_kmeans_model(descriptors, dictionarySize)
    print(len(indexes))
    print(len(descriptors))
    print(model.cluster_centers_)
    with open('model.sift.pickle', 'wb') as f:
        pickle.dump(model, f)
    return model

#Jaccard Similarities
# Calculate and store the actual Jaccard similarity.
#JSim[getTriangleIndex(i, j)] = (len(s1.intersection(s2)) / len(s1.union(s2)))

#MinHash Signatures
maxWordID = dictionarySize# Size of vocabulary assigned.
# next largest prime number above maxWordID http://compoasso.free.fr/primelistweb/page/prime/liste_online_en.php
nextPrime = 7#4294967311

def pickRandomCoeffs(k):
  randList = []# Create a list of 'k' random values.
  while k > 0:
    randIndex = random.randint(0, maxWordID)# Get a random word ID.
    while randIndex in randList:# Ensure that each random number is unique.
      randIndex = random.randint(0, maxWordID)
    randList.append(randIndex)# Add the random number to the list.
    k = k - 1
return randList
# For each of the 'numHashes' hash functions, generate a different coefficient 'a' and 'b'.
coeffA = pickRandomCoeffs(numHashes)
coeffB = pickRandomCoeffs(numHashes)

# List of images represented as signature vectors
signatures = []
sketches = []
for imID in range(0,numImages):# For each image
  bowIDSet = bagOfWords[imID]# Get the bag of words for this image
  signature = []  # The resulting minhash signature for this image
  for i in range(0, numHashes):  # For each of the random hash functions:
    # For every word in the image, calculate its hash code using hash function 'i'.
    # Track the lowest hash ID seen. Initialize 'minHashCode' to be greater than
    # the maximum possible value output by the hash.
    minHashCode = nextPrime + 1
    for wordID in bowIDSet:# For each word in the image
      hashCode = (coeffA[i] * wordID + coeffB[i]) % nextPrime# compute hash function
      if hashCode < minHashCode:
          minHashCode = hashCode#retain the lowest hash code
          signature.append(minHashCode)# Add the smallest hash code as component 'i' of signature
          sketchPosition = wordID % lenSketches
          sketch[(len(signature)-1),sketchPosition] = minHashCode# COMPUTE SKETCHES N=k*n N hashes, k sketches with length n
signatures.append(signature)# Store the MinHash signature for this image.

# COMPUTE SKETCHES N=k*n N hashes, k sketches with length n
sketches = []
for i in range(0,numImages):
    sketch = []
    sketch.append(currentSketch)
sketches.append(sketch)

# COMPARE SKETCHES - require at least one match - add later input parameter m
matches = []
for i in range(0,numSketches):
    match = 0
    if sketch1(i)=sketch2(i):
    match =+ 1
    matches.append(match)
numMatches = len(matches)

#LEN(MATCHES)/LEN(SKETCHES)>35% - get index
# COMPARE HASHES FOR PAIRS WITH MIN MATCHING SKETCHES
# For each of the test images
matchedImages = []
for i in range(0, numMatches):
  signature1 = signatures[i]# Get the MinHash signature for image i.
  for j in range(i + 1, numMatches):  # For each of the other test images
    signature2 = signatures[j]# Get the MinHash signature for image j.
    count = 0
    # Count the number of positions in the minhash signature which are equal.
    for k in range(0, numHashes):
        count = count + (signature1[k] == signature2[k])
        # Record the percentage of positions which matched.
        matchedImages(i, j) = (count / numHashes)

# LIST MATCHING IMAGES WITH SCORES
# For each of the test images
rankedMatches = []
indMatchedImages = find(matchedImages)
# sort match SCORES
rankedIndexes = sort(matchedImages(indMatchedImages))
# Sort them in the order of their distance.
#matches = sorted(matches, key = lambda x:x.distance)
