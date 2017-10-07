import os
import utils
import numpy as np

from keras.utils import Sequence
from skimage import io, transform
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


def generate_variant(fname):
    lot_weather_date_occupied = fname.split('/')[-5:-1]
    lot_weather_date_occupied.pop(2)
    return '_'.join(lot_weather_date_occupied)


class Dataset(object):

    def __init__(self, base_dir='../data'):
        self._base_dir = base_dir
        self._dir = os.path.join(base_dir, 'PKLotSegmented')

        self.X = np.array(utils.recursive_glob(self._dir, "*.jpg"))
        self.y = np.array(["Occupied" in s for s in self.X]).astype(np.float32)
        self.variant = np.array([generate_variant(s) for s in self.X])

        self._test_indices = None
        self._train_indices = None

    @property
    def test_indices(self):
        if self._test_indices is None:
            save_path = os.path.join(self._base_dir, 'seg_test_indices.npy')
            if os.path.exists(save_path):
                self._test_indices = np.load(save_path)
                self._train_indices = np.setdiff1d(np.arange(0, len(self.X)), self._test_indices)
            else:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
                self._train_indices, self._test_indices = next(sss.split(np.zeros(len(self.variant)), self.variant))
                np.save(save_path, self._test_indices)
        return self._test_indices

    def train_val_split(self, val_size=0.1):
        # Ensure train/test split.
        _ = self.test_indices

        return train_test_split(self.X[self._train_indices], self.y[self._train_indices],
                                test_size=val_size,
                                stratify=self.y[self._train_indices])

    def get_test_data(self):
        return self.X[self.test_indices], self.y[self.test_indices]


class DataLoader(Sequence):

    def __init__(self, X, y, img_width=64, img_height=64, batch_size=32):
        shuffled_indices = np.arange(len(X))
        np.random.shuffle(shuffled_indices)
        self.X = X[shuffled_indices]
        self.y = y[shuffled_indices]

        self.img_width = img_width
        self.img_height = img_height
        self.batch_size = batch_size

    def __len__(self):
        return len(self.X) // self.batch_size

    def __getitem__(self, batch_idx):
        X = self.X[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
        y = self.y[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]

        batch_x = np.zeros(shape=(len(X), self.img_height, self.img_width, 3), dtype=np.float32)
        batch_y = np.zeros(shape=len(X), dtype=np.float32)

        for i in range(len(X)):
            img = utils.random_rotate_45(io.imread(X[i]))
            batch_x[i] = transform.resize(img, output_shape=(self.img_height, self.img_width),
                                          preserve_range=True).astype('uint8')
            batch_y[i] = y[i]

        return utils.center(batch_x), batch_y


if __name__ == '__main__':
    ds = Dataset()
    dl = DataLoader(ds.X, ds.y)
    x, y = dl[0]
