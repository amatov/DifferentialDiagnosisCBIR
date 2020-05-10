# Based on an image dataset, compute useful statistics such
# as the average number of feature per image for both SIFT and SURF,
# average time to detect/compute SIFT and SURF, etc.

import itertools
import glob
import os
import cv2
import time
import pprint
from math import ceil
from tqdm import tqdm
from numpy import average, median
import multiprocessing as mp

sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()

def image_filenames(basepath, limit = None):
    files_iter = glob.iglob(os.path.join(basepath, '*.jpg'))
    return itertools.islice(files_iter, 0, limit)

def surf_count(image, filename):
    kp, des = surf.detectAndCompute(image, None)
    return len(kp)

def sift_count(image, filename):
    kp, des = sift.detectAndCompute(image, None)
    return len(kp)

def analyze_image(image, filename):
    analyze_fns = {
            'filename': lambda image, filename: filename,
            'sift_count': sift_count,
            'surf_count': surf_count,
    }
    results = {}
    for key, fn in analyze_fns.items():
        start = time.time()
        results[key] = fn(image, filename)
        end = time.time()
        delta = end - start
        results[key + '_timing'] = delta
    return results

def compute_stats(data):
    # e.g. pick('a', [{'a': 1},{'a': 3}]) === [1, 3]
    def pick(key, lst): return list(map(lambda x: x[key], lst))

    stat_fns = {
            'avg': average,
            'median': median,
    }

    stats = {}

    for algo in ['sift', 'surf']:
        stats[algo] = {
                'feature_count': {},
                'time': {},
        }
        s = stats[algo]
        feature_counts = pick(algo + '_count', data)
        timings = pick(algo + '_count_timing', data)
        for key, fn in stat_fns.items():
            s['feature_count'][key] = fn(feature_counts)
            s['time'][key] = fn(timings)
    return stats

def analyze_mapper(filename):
    image = cv2.imread(filename)
    return analyze_image(image, filename)

def analyze(basepath, limit, cpus, progress_fn = lambda x: x, results = []):
    filenames = list(image_filenames(basepath, limit))
    total = len(filenames)
    with mp.Pool(processes = cpus) as p:
        chunk_size = 1
        for val in p.imap_unordered(analyze_mapper, filenames, chunk_size):
            results.append(val)
            progress_fn(total)
    return results

def create_progress_fn():
    memo = {
        'pbar': None,
        'i': 0,
    }
    def p(total):
        # print('%d / %d' % (i, total))
        if memo['pbar'] is None:
            pbar = tqdm(
                    total=total,
                    desc='Analyzing',
                    unit='image',
            )
            memo['pbar'] = pbar
        pbar = memo['pbar']
        pbar.update(1)
        memo['i'] += 1
        if (memo['i'] == total):
            pbar.close()
    return p

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
    # print(data)
    stats = compute_stats(data)
    pprint.pprint(stats, width = 1)
    print('')
    print('PS: time is in seconds')
    print('')
    print('On average,')
    print('')
    print('* An image has %d SIFT features' % stats['sift']['feature_count']['avg'])
    print('* An image has %d SURF features' % stats['surf']['feature_count']['avg'])
    print('* It takes %.2f seconds to detect and compute SIFT features' % stats['sift']['time']['avg'])
    print('* It takes %.2f seconds to detect and compute SURF features' % stats['surf']['time']['avg'])
    print(
            '* SIFT finds %.1fx as many features as SURF'
            %
            (stats['sift']['feature_count']['avg'] / stats['surf']['feature_count']['avg'])
            )
    print(
            '* SIFT takes %.1fx as long as SURF to detect and compute descriptors'
            %
            (stats['sift']['time']['avg'] / stats['surf']['time']['avg'])
            )
    print('')

if __name__ == '__main__':
    main()
