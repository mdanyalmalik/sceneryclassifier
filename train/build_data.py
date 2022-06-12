import cv2
import numpy as np
import os
from tqdm import tqdm

REBUILD_DATA = False  # Set to true to rebuild data

trainset_dir = "data/seg_train/"
testset_dir = "data/seg_test/"


class DataBuilder():
    IMG_SIZE = 150

    BUILDINGS_TRAIN, BUILDINGS_TEST = trainset_dir + \
        "buildings", testset_dir+"buildings"
    FOREST_TRAIN, FOREST_TEST = trainset_dir + "forest", testset_dir+"forest"
    GLACIER_TRAIN, GLACIER_TEST = trainset_dir + "glacier", testset_dir+"glacier"
    MOUNTAIN_TRAIN, MOUNTAIN_TEST = trainset_dir + "mountain", testset_dir+"mountain"
    SEA_TRAIN, SEA_TEST = trainset_dir + "sea", testset_dir+"sea"
    STREET_TRAIN, STREET_TEST = trainset_dir + "street", testset_dir+"street"

    LABELS_TRAIN = {BUILDINGS_TRAIN: 0, FOREST_TRAIN: 1,
                    GLACIER_TRAIN: 2, MOUNTAIN_TRAIN: 3, SEA_TRAIN: 4, STREET_TRAIN: 5}
    LABELS_TEST = {BUILDINGS_TEST: 0, FOREST_TEST: 1,
                   GLACIER_TEST: 2, MOUNTAIN_TEST: 3, SEA_TEST: 4, STREET_TEST: 5}

    training_data = []
    testing_data = []

    def make_training_data(self):
        for label in self.LABELS_TRAIN:
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    self.training_data.append(
                        [np.array(img), np.eye(6)[self.LABELS_TRAIN[label]]])
                except Exception as e:
                    print(e)

        np.random.shuffle(self.training_data)
        np.save("data/training_data.npy", self.training_data)

    def make_testing_data(self):
        for label in self.LABELS_TEST:
            for f in tqdm(os.listdir(label)):
                try:
                    path = os.path.join(label, f)
                    img = cv2.imread(path)
                    img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                    self.testing_data.append(
                        [np.array(img), np.eye(6)[self.LABELS_TEST[label]]])
                except Exception as e:
                    print(e)

        np.random.shuffle(self.testing_data)
        np.save("data/testing_data.npy", self.testing_data)


if REBUILD_DATA:
    data_builder = DataBuilder()
    data_builder.make_training_data()
    data_builder.make_testing_data()
