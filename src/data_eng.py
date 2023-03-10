####################################################################################################################################
####################################################################################################################################
#############################################################LIBRARIES##############################################################

import pandas as pd
import numpy as np
import os
import cv2

####################################################################################################################################
####################################################################################################################################
######################################################FUNCTIONS AND CLASSES#########################################################

####################################################################################################################################
# Function that calculates frequencies of elements in a list:

class ImportCleanImages:
    """
    Class that imports and prepares data from images.

    Initialization parameters:
      :param path_to_files: relative path to the files (raw images and names of files in training, validation and test datasets).
      :type path_to_files: string.
      :param images_folder: name of the folder where raw images can be found.
      :type images_folder: string.
      :param train_val_test_file: name of files indicating if each file belongs to training, validation or test datasets.
      :type train_val_test_file: string.
      :param label_dict: dictionary providing an integer label for each class of images.
      :type label_dict: dictionary.
      :param grey_scale: declares whether images should be read in grey scale or RGB.
      :type grey_scale: boolean.
      :param shuffle: declares whether data should be shuffled.
      :type shuffle: boolean.
      :param resize: declares whether to resize images.
      :type resize: boolean.
      :param width: width for resizing images.
      :type width: integer.
      :param height: height for resizing images.
      :type height: integer.
      :param interpolation: interpolation method for resizing images.
      :type interpolation: string.
      :param seed: fixes the shuffling of images in each dataset.
      :type seed: integer.

    Methods:
      "build_datasets": method that implements all necessary operations (coded in protected methods) for importing and cleaning raw
      images. Returns a dictionary whose keys are the indication of training, validation or test datasets, and whose values are tuples
      with matrices of pixels, vectors of labels, and the identification of each image, respectively.
    """
    def __str__(self):
        params = ', '.join(
            [f'{k}={v}' for k, v in self.__dict__.items()]
            )
        return f'{self.__class__.__name__}({params})'

    def __repr__(self):
        return self.__str__()
    
    def __init__(self, path_to_files: str, label_dict: dict,
                 images_folder: str = "images", train_val_test_file: str = "images",
        				 grey_scale: bool = False, shuffle: bool = True,
                 resize: bool = False, width: int = 224, height: int = 224,
                 interpolation: str = "default", seed: int = 1):
      self.path_to_files = path_to_files
      self.images_folder = images_folder
      self.train_val_test_file = train_val_test_file
      self.label_dict = label_dict
      self.grey_scale = grey_scale
      self.shuffle = shuffle
      self.classes = os.listdir(f'{path_to_files}/{images_folder}') # Collection of labels.
      self.resize = resize
      self.width = width
      self.height = height
      self.interpolation = interpolation
      self.seed = seed

    def build_datasets(self) -> dict:
        # Importing the names (identification) of images:
        self.__import_names()

        # Importing and processing the images:
        images_train, images_val, images_test, labels_train, labels_val, labels_test,\
          ids_train, ids_val, ids_test = self.__import_images()

        # Final processing of images:
        images_train, images_val, images_test, labels_train, labels_val, labels_test,\
          ids_train, ids_val, ids_test = self.__cleaning_images(
            images_train, images_val, images_test, labels_train, labels_val, labels_test, ids_train, ids_val, ids_test
            )

        return {'train': (images_train, labels_train, ids_train),
                'val': (images_val, labels_val, ids_val),
                'test': (images_test, labels_test, ids_test)}

    def __import_names(self):
        # File names of each dataset:
        self.train_images = [f.split('.')[0] for f in list(pd.read_csv(f'{self.path_to_files}/train_{self.train_val_test_file}.txt',
                                          sep=' ', header=None)[0])]
        self.val_images = [f.split('.')[0] for f in list(pd.read_csv(f'{self.path_to_files}/val_{self.train_val_test_file}.txt',
                                          sep=' ', header=None)[0])]
        self.test_images = [f.split('.')[0] for f in list(pd.read_csv(f'{self.path_to_files}/test_{self.train_val_test_file}.txt',
                                          sep=' ', header=None)[0])]

    def __import_images(self):
        images_train, images_val, images_test = {}, {}, {}
        labels_train, labels_val, labels_test = {}, {}, {}
        ids_train, ids_val, ids_test = {}, {}, {}

        # Loop over classes:
        for c in self.classes:
            # Training data:
            images_train[c] = dict(
              zip(
                  # Identification of images:
                  [i.split('.')[0] for i in os.listdir(f'{self.path_to_files}/{self.images_folder}/{c}') if
                    i.split('.')[0] in self.train_images],
                  # Reading the images:
                  [
                    cv2.imread(f'{self.path_to_files}/{self.images_folder}/{c}/{i}', 0) if self.grey_scale else \
                      cv2.cvtColor(cv2.imread(f'{self.path_to_files}/{self.images_folder}/{c}/{i}', -1), cv2.COLOR_BGR2RGB)
                    for i in os.listdir(f'{self.path_to_files}/{self.images_folder}/{c}') if
                     i.split('.')[0] in self.train_images
                  ]
              )
            )

            ids_train[c] = list(images_train[c].keys()) # Identification of images.
            images_train[c] = np.stack([images_train[c][i] for i in images_train[c]], axis=0) # Collection of images.
            labels_train[c] = [self.label_dict[c] for i in range(len(images_train[c]))] # Labels of images.

            # Validation data:
            images_val[c] = dict(
              zip(
                  # Identification of images:
                  [i.split('.')[0] for i in os.listdir(f'{self.path_to_files}/{self.images_folder}/{c}') if
                    i.split('.')[0] in self.val_images],
                  # Reading the images:
                  [
                    cv2.imread(f'{self.path_to_files}/{self.images_folder}/{c}/{i}', 0) if self.grey_scale else \
                      cv2.cvtColor(cv2.imread(f'{self.path_to_files}/{self.images_folder}/{c}/{i}', -1), cv2.COLOR_BGR2RGB)
                    for i in os.listdir(f'{self.path_to_files}/{self.images_folder}/{c}') if
                     i.split('.')[0] in self.val_images
                  ]
              )
            )

            ids_val[c] = list(images_val[c].keys()) # Identification of images.
            images_val[c] = np.stack([images_val[c][i] for i in images_val[c]], axis=0) # Collection of images.
            labels_val[c] = np.array([self.label_dict[c] for i in range(len(images_val[c]))]) # Labels of images.

            # Test data:
            images_test[c] = dict(
              zip(
                  # Identification of images:
                  [i.split('.')[0] for i in os.listdir(f'{self.path_to_files}/{self.images_folder}/{c}') if
                    i.split('.')[0] in self.test_images],
                  # Reading the images:
                  [
                    cv2.imread(f'{self.path_to_files}/{self.images_folder}/{c}/{i}', 0) if self.grey_scale else \
                      cv2.cvtColor(cv2.imread(f'{self.path_to_files}/{self.images_folder}/{c}/{i}', -1), cv2.COLOR_BGR2RGB)
                    for i in os.listdir(f'{self.path_to_files}/{self.images_folder}/{c}') if
                     i.split('.')[0] in self.test_images
                  ]
              )
            )

            ids_test[c] = list(images_test[c].keys()) # Identification of images.
            images_test[c] = np.stack([images_test[c][i] for i in images_test[c]], axis=0) # Collection of images.
            labels_test[c] = [self.label_dict[c] for i in range(len(images_test[c]))] # Labels of images.

        return images_train, images_val, images_test, labels_train, labels_val, labels_test, ids_train, ids_val, ids_test

    def __cleaning_images(self, images_train, images_val, images_test, labels_train, labels_val, labels_test,
                          ids_train, ids_val, ids_test):
        # Training data:
        ids_train = [item for sublist in [ids_train[c] for c in ids_train] for item in sublist] # Identification of images.
        images_train = np.concatenate([images_train[c] for c in images_train], axis=0) # Collection of images.
        labels_train = [item for sublist in [labels_train[c] for c in labels_train] for item in sublist] # Labels of images.

        # Validation data:
        ids_val = [item for sublist in [ids_val[c] for c in ids_val] for item in sublist] # Identification of images.
        images_val = np.concatenate([images_val[c] for c in images_val], axis=0) # Collection of images.
        labels_val = [item for sublist in [labels_val[c] for c in labels_val] for item in sublist] # Labels of images.

        # Test data:
        ids_test = [item for sublist in [ids_test[c] for c in ids_test] for item in sublist] # Identification of images.
        images_test = np.concatenate([images_test[c] for c in images_test], axis=0) # Collection of images.
        labels_test = [item for sublist in [labels_test[c] for c in labels_test] for item in sublist] # Labels of images.

        if self.shuffle:
            train_shuffle, val_shuffle, test_shuffle = self.__shuffling(images_train, images_val, images_test)

            # Training data:
            images_train = images_train[train_shuffle, :, :, :] if self.grey_scale==False else images_train[train_shuffle, :, :]
            labels_train = [labels_train[i] for i in train_shuffle]
            ids_train = [ids_train[i] for i in train_shuffle]

            # Validation data:
            images_val = images_val[val_shuffle, :, :, :] if self.grey_scale==False else images_val[val_shuffle, :, :]
            labels_val = [labels_val[i] for i in val_shuffle]
            ids_val = [ids_val[i] for i in val_shuffle]

            # Test data:
            images_test = images_test[test_shuffle, :, :, :] if self.grey_scale==False else images_test[test_shuffle, :, :]
            labels_test = [labels_test[i] for i in test_shuffle]
            ids_test = [ids_test[i] for i in test_shuffle]

        if self.resize:
            images_train = np.array(
              [self.__resize(img, width=self.width, height=self.height, interpolation=self.interpolation) for img in images_train]
            )
            images_val = np.array(
              [self.__resize(img, width=self.width, height=self.height, interpolation=self.interpolation) for img in images_val]
            )
            images_test = np.array(
              [self.__resize(img, width=self.width, height=self.height, interpolation=self.interpolation) for img in images_test]
            )

        return images_train, images_val, images_test, labels_train, labels_val, labels_test, ids_train, ids_val, ids_test

    def __shuffling(self, images_train, images_val, images_test):
        # Shuffling the indexes:
        np.random.seed(self.seed)
        train_shuffle = list(np.random.choice(range(len(images_train)), size=len(images_train), replace=False))
        np.random.seed(self.seed)
        val_shuffle = list(np.random.choice(range(len(images_val)), size=len(images_val), replace=False))
        np.random.seed(self.seed)
        test_shuffle = list(np.random.choice(range(len(images_test)), size=len(images_test), replace=False))

        return train_shuffle, val_shuffle, test_shuffle

    def __resize(self, image: np.ndarray, width: int = 224, height: int = 224, interpolation: str = 'default') -> np.ndarray:
        if interpolation=='default':
            interp = cv2.INTER_AREA

        return cv2.resize(image, (width, height), interpolation=interp)
