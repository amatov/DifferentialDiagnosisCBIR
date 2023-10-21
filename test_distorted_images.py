from imgaug import augmenters as iaa
import cv2
from scipy import ndimage, misc

seq = iaa.Sequential([
    iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 2.0)) # blur images with a sigma of 0 to 3.0
])


    # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
    # or a list of 3D numpy arrays, each having shape (height, width, channels).
    # Grayscale images must have shape (height, width, 1) each.
    # All images must have numpy's dtype uint8. Values are expected to be in
    # range 0-255.

images = [ndimage.imread('/media/sf_flickr-images/im20014.jpg')]
print(images)
images_aug = seq.augment_images(images)

i = 3
for image in images_aug:
    i += 1
    misc.imsave('/media/sf_flickr-images/surfer_distortion_%s_.jpg'%i,image)
    #train_on_images(images_aug)
