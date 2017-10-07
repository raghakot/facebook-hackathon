import os
import fnmatch
import numpy as np

from datetime import datetime
from skimage.transform import rotate


def recursive_glob(dir='.', pattern='*'):
    """Search recursively for files matching a specified pattern.
    http://stackoverflow.com/questions/2186525/use-a-glob-to-find-files-recursively-in-python
    """
    matches = []
    for root, dirnames, filenames in os.walk(dir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches


def generate_run_id():
    return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


def center(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x


def decode(x):
    return ((x + 1.0) * 0.5 * 255).astype(np.uint8)


def random_rotate_45(x):
    idx = np.random.randint(0, 8)
    return rotate(x, idx * 45, center=None, resize=True, preserve_range=True).astype(np.uint8)
