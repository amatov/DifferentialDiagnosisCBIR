# Dependencies

No `requirements.txt` is included. The scripts import the following
third-party Python packages:

- **opencv-contrib-python** -- required for `cv2.xfeatures2d.SIFT_create()`
  and `cv2.xfeatures2d.SURF_create()`. SURF is patent-encumbered and is
  only available in older opencv-contrib-python releases (<= 3.4.2.16
  reliably provides it without a special build flag); SIFT became
  patent-free and moved into the main `opencv-python` package starting
  with OpenCV 4.4. Scripts using SURF specifically require an older
  opencv-contrib-python version.
- **elasticsearch** (the Python client) -- the `query_*_es.py` and
  `test_*_es*.py` scripts connect to a running ElasticSearch instance
  (localhost:9200 by default); ElasticSearch itself is not included and
  must be installed and running separately.
- **scikit-learn** -- used for k-means clustering (`sklearn.cluster`) and
  PCA (`sklearn.decomposition.pca`).
- **scipy** -- `test_distorted_images.py` uses `scipy.ndimage.imread` and
  `scipy.misc.imsave`, both of which were removed in modern scipy
  releases (deprecated since scipy 1.0, removed by 1.3); an older scipy
  version or an updated call using `imageio`/`Pillow` is needed to run
  this script as-is.
- **imgaug** -- used for image augmentation in `test_distorted_images.py`.
- **numpy**, **matplotlib**, **tqdm** -- general array handling,
  plotting, and progress bars.

## Missing input data

Several scripts reference relative image-dataset folders that are not
included in this repository and must be supplied separately:
`./flickr-images/`, `./1000testImages/`, `./images/moreImages/`, and
`./out/` (for saved model/output files). A few scripts also reference an
absolute path from the original author's virtual machine
(`/media/sf_flickr-images/`, a VirtualBox shared folder); these are
flagged with `# EDIT:` comments in the source.
