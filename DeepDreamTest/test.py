import os
import numpy as np
import scipy.ndimage as nd
import PIL.Image

import kernel

def showAvailableLayers():
    """
    Read the function title
    The net is the one load in the kernel module
    """
    kernel.net.blobs.keys()

def dream(img_path, end='inception_3b/5x5_reduce'):
    """
    Simple dream on a image
    """
    img = np.float32(PIL.Image.open(img_path))
    showarray(img)
    _=kernel.deepdream(kernel.net, img, end=end)

def recursive_dream(img_path, end='inception_4c/output', rec=10):
    """
    Apply the dream on the img_path and reuse the output for dreaming again
    rec: number of times reusing the output
    end: layer selection
    """
    # Create repository where to save pictures
    os.system('mkdir frames')
    frame = np.float32(PIL.Image.open(img_path))
    frame_i = 0

    # Let's dream
    h, w = frame.shape[:2]
    s = 0.05 # scale coefficient
    for i in xrange(rec):
        frame = deepdream(kernel.net, frame, end=end)
        PIL.Image.fromarray(np.uint8(frame)).save("frames/%04d.jpg"%frame_i)
        frame = nd.affine_transform(frame, [1-s,1-s,1], [h*s/2,w*s/2,0], order=1)
        frame_i += 1

def guide_dream(img_path, guide_path, end='inception_3b/output'):
    """
    Apply a dream by following a specific pattern from the image "guide_path"
    """
    img = np.float32(PIL.Image.open(img_path))
    dst, guide_features = setGuide(guide_path, end=end)
    _=kernel.deepdream(kernel.net, img, end=end, objective=objective_guide)

def setGuide(img_path, end = 'inception_3b/output'):
    """
    Extract the good features for guiding the dreaming
    """
    guide = np.float32(PIL.Image.open(img_path))

    h, w = guide.shape[:2]
    src, dst = kernel.net.blobs['data'], kernel.net.blobs[end]
    src.reshape(1,3,h,w)
    src.data[0] = kernel.preprocess(kernel.net, guide)
    kernel.net.forward(end=end)
    guide_features = dst.data[0].copy()

    return dst, guide_features
