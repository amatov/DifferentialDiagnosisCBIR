import numpy
def distance(desc1, desc2):
    dist = numpy.linalg.norm(desc1-desc2)
    return dist

def searchvisualword(dictionary,descriptor):
    minDistance = distance(dictionary[0], descriptor)
    closestCenter = 0
    for idx, clusterCenter in enumerate(dictionary):
        dis = distance(clusterCenter, descriptor)
        if dis < minDistance:
            minDistance = dis
            closestCenter = idx
    return closestCenter

def getHistogram(dictionary, des):
    histo = [0,0,0,0,0]
    for descriptor in des:
        word = searchvisualword(dictionary,descriptor)
        histo[word] += 1
    return histo
