####################################################################################################################################
####################################################################################################################################
#############################################################LIBRARIES##############################################################

import numpy as np
import cv2
from abc import ABC, abstractmethod
from copy import deepcopy

from keras.preprocessing.image import ImageDataGenerator

from transformations import brightness_filter

####################################################################################################################################
####################################################################################################################################
#######################################################FUNCTIONS AND CLASSES########################################################

####################################################################################################################################
# Abstract class that defines the structure of classes of image transformation:

class ImageOperation(ABC):
    def __str__(self):
        params = ', '.join(
            [f'{k}={v}' for k, v in self.__dict__.items()]
            )
        return f'{self.__class__.__name__}({params})'

    def __repr__(self):
        return self.__str__()

    def transform(self, image: np.ndarray) -> np.ndarray:
        pass

####################################################################################################################################
# Class to randomly change the brightness of images:

class Brightness(ImageOperation):
    """
    Class to randomly change the brightness of images.

    Arguments for initialization:
        :param change_range: interval from where a proportion of change in brightness is randomly
        picked.
        :type change_range: boolean.
        :param bright_prob: probability under which the change in brightness is implemented.
        :type bright_prob: float.
    """
    def __init__(self, change_range: tuple, bright_prob: float = 1.0):
        self.change_range = change_range
        self.bright_prob = bright_prob

    def transform(self, image: np.ndarray) -> np.ndarray:
        if np.random.choice([True, False], size=1, replace=False, p=[self.bright_prob, 1-self.bright_prob])[0]:
            random_delta = round(np.random.uniform(low=self.change_range[0], high=self.change_range[1], size=1)[0], 2)
            return self.change_brightness(image=image, delta=random_delta)
        else:
            return image

    @staticmethod
    def change_brightness(image: np.ndarray, delta: float) -> np.ndarray:
        """
        Function for changing the overall brightness of an image.
        
        :param image: original image whose brightness should be changed.
        :type image: nd-array.
        :param delta: proportion of change in brightness to be imposed on the image. delta = 1 means doubling the
        overall brightness, while negative values imply in a reduction.
        in brightness.
        :type delta: float.

        :return: modified image.
        :rtype: nd-array.
        """
        return brightness_filter(image=image, delta=delta)

####################################################################################################################################
# Class to randomly flip images:

class Flip(ImageOperation):
    """
    Class to randomly flip images.

    Arguments for initialization:
        :param horizontal: indicates whether to implement horizontal flip.
        :type horizontal: boolean.
        :param vertical: indicates whether to implement vertical flip.
        :type vertical: boolean.
        :param flip_prob: probability under which the flip operation is implemented.
        :type flip_prob: float.
    """
    def __init__(self, horizontal: bool = True, vertical: bool = False, flip_prob: float = 0.5):
        self.horizontal = horizontal
        self.vertical = vertical
        self.flip_prob = flip_prob

    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Method for implementing random flip operation.

        :param image: original image to be randomly flipped.
        :type image: nd-array.

        :return: flipped image.
        :rtype: nd-array.
        """
        if np.random.choice([True, False], size=1, replace=False, p=[self.flip_prob, 1-self.flip_prob])[0]:
            return self.flip(image=image, horizontal=self.horizontal, vertical=self.vertical)
        else:
            return image

    @staticmethod
    def flip(image: np.ndarray, horizontal: bool = True, vertical: bool = False) -> np.ndarray:
        """
        Function that implements flip operation over images.

        :param image: original image to be flipped.
        :type image: nd-array.
        :param horizontal: indicates whether to implement horizontal flip.
        :type horizontal: boolean.
        :param vertical: indicates whether to implement vertical flip.
        :type vertical: boolean.

        :return: flipped image.
        :rtype: nd-array.
        """
        if horizontal & vertical:
            return cv2.flip(image, -1)

        elif horizontal:
            return cv2.flip(image, 1)

        elif vertical:
            return cv2.flip(image, 0)
        
        else:
          return

####################################################################################################################################
# Class to randomly rotate images:

class Rotation(ImageOperation):
    """
    Class to randomly rotate images.

    Arguments for initialization:
        :param rotation_range: range of values from where an angle is going to be
        picked to rotate the image.
        :type rotation_range: tuple.
        :param rotation_prob: probability under which the rotation operation is implemented.
        :type rotation_prob: float.
    """
    def __init__(self, rotation_range: tuple, rotation_prob: float = 1.0):
        self.rotation_range = rotation_range
        self.rotation_prob = rotation_prob

    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Method for implementing random rotation.

        :param image: original image to be randomly rotated.
        :type image: nd-array.

        :return: rotated image.
        :rtype: nd-array.
        """
        if np.random.choice([True, False], size=1, replace=False, p=[self.rotation_prob, 1-self.rotation_prob])[0]:
            random_angle = np.random.choice([i for i in range(self.rotation_range[0], self.rotation_range[1]+1)],
                                            size=1, replace=False)[0]
            return self.rotation(image=image, angle=random_angle)
        else:
            return image

    @staticmethod
    def rotation(image: np.ndarray, angle: float) -> np.ndarray:
        """
        Function that implements rotation operation over images.

        :param image: original image to be rotated.
        :type image: nd-array.
        :param angle: angle for the rotation.
        :type angle: float.

        :return: rotated image.
        :rtype: nd-array.
        """
        # Original dimensions:
        height, width, channels = image.shape
        center = (height//2, width//2)

        # Rotation matrix:
        M = cv2.getRotationMatrix2D(center, angle=angle, scale=1.0)

        # Rotated image:
        return cv2.warpAffine(image, M, (width, height))

####################################################################################################################################
# Class to randomly shift images:

class Shift(ImageOperation):
    """
    Class to randomly shift images.

    Arguments for initialization:
        :param tx_range: range of values for the amount of pixels that the image should go to the right.
        :type tx_range: tuple.
        :param ty_range: range of values for the amount of pixels that the image should go to the bottom.
        :type ty_range: tuple.
        :param shift_prob: probability under which the shift operation is implemented.
        :type shift_prob: float.
    """
    def __init__(self, tx_range: tuple, ty_range: tuple, shift_prob: float = 1.0):
        self.tx_range = tx_range
        self.ty_range = ty_range
        self.shift_prob = shift_prob

    def transform(self, image: np.ndarray) -> np.ndarray:
        if np.random.choice([True, False], size=1, replace=False, p=[self.shift_prob, 1-self.shift_prob])[0]:
            random_tx = np.random.choice([i for i in range(self.tx_range[0], self.tx_range[1]+1)],
                                         size=1, replace=False)[0]
            random_ty = np.random.choice([i for i in range(self.ty_range[0], self.ty_range[1]+1)],
                                         size=1, replace=False)[0]
            return self.shift(image=image, tx=random_tx, ty=random_ty)
        else:
            return image

    @staticmethod
    def shift(image: np.ndarray, tx: int, ty: int) -> np.ndarray:
        """
        Function that produces horizontal and vertical translation of images inside their plots.

        :param image: original image to be shifted.
        :type image: nd-array.
        :param tx: amount of pixels that the image should go to the right. A negative number will move the image to the left.
        :type tx: int.
        :param ty: amount of pixels that the image should go to the bottom. A negative number will move the image to the top.
        :type ty: int.

        :return: translated image.
        :rtype: nd-array.
        """
        # Original dimensions:
        height, width, channels = image.shape

        # Translation matrix:
        M = np.float32([[1,0,tx],[0,1,ty]])

        # Shifted image:
        return cv2.warpAffine(image, M, (width, height))

####################################################################################################################################
# Class to randomly crop images:

class Crop(ImageOperation):
    """
    Class to randomly crop images.

    Arguments for initialization:
        :param center_range: interval from where a random value of crop center will be picked.
        :type center_range: tuple.
        :param window_range: interval from where a random value of crop window will be picked.
        :type window_range: tuple.
        :param crop_prob: probability under which the crop operation is implemented.
        :type crop_prob: float.
    """
    def __init__(self, center_range: tuple, window_range: tuple, crop_prob: float = 1.0):
        self.center_range = center_range
        self.window_range = window_range
        self.crop_prob = crop_prob

    def transform(self, image: np.ndarray) -> np.ndarray:
        if np.random.choice([True, False], size=1, replace=False, p=[self.crop_prob, 1-self.crop_prob])[0]:
            random_center = round(np.random.uniform(low=self.center_range[0], high=self.center_range[1], size=1)[0], 2)
            random_window = np.random.choice([i for i in range(self.window_range[0], self.window_range[1]+1)],
                                             size=1, replace=False)[0]
            return self.crop(image=image, crop_center=random_center, crop_window=random_window)
        else:
            return image

    @staticmethod
    def crop(image: np.ndarray, crop_center: float = 0.5, crop_window: int = 100) -> np.ndarray:
        """
        Function that generates a cropped version of an image.

        :param image: original image to be shifted.
        :type image: nd-array.
        :param crop_center: defines the point in the image that will be the center of the cropped image.
        :type crop_center: float.
        :param crop_window: defines the subset of the image to be cropped in. The smaller this parameter,
        the bigger the zoom-in.
        :type crop_window: integer.

        :return: translated image.
        :rtype: nd-array.
        """
        # Original dimensions:
        height, width, channels = image.shape

        # Cropped dimensions:
        cropped_height = int(height*crop_center)
        cropped_width = int(width*crop_center)

        # Cropped image:
        cropped_img = image[cropped_height-crop_window: cropped_height+crop_window,
                            cropped_width-crop_window: cropped_width+crop_window, :]

        # Resized image:
        return cv2.resize(cropped_img, (width, height), interpolation=cv2.INTER_AREA)

####################################################################################################################################
# Class to randomly implement blur filtering over images:

class Blur(ImageOperation):
    """
    Class to randomly implement blur filtering over images.

    Arguments for initialization:
        :param kernel_size: dimensions of the kernel matrix for the filtering.
        :type kernel_size: tuple.
        :param blur_prob: probability under which the blur filtering is implemented.
        :type blur_prob: float.
    """
    def __init__(self, kernel_size: tuple, blur_prob: float = 0.5):
        self.kernel_size = kernel_size
        self.blur_prob = blur_prob

    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Method for implementing random blur.

        :param image: original image to be randomly blurred.
        :type image: nd-array.

        :return: blurred image.
        :rtype: nd-array.
        """
        if np.random.choice([True, False], size=1, replace=False, p=[self.blur_prob, 1-self.blur_prob])[0]:
            return self.blur(image=image, kernel_size=self.kernel_size)
        else:
            return image

    @staticmethod
    def blur(image: np.ndarray, kernel_size: tuple) -> np.ndarray:
        """
        Function that implements blur filter operation over images.

        :param image: original image to be rotated.
        :type image: nd-array.
        :param kernel_size: dimensions of the kernel matrix for the filtering.
        :type kernel_size: tuple.

        :return: rotated image.
        :rtype: nd-array.
        """
        return cv2.blur(image, kernel_size)

####################################################################################################################################
# Class that augments a batch of original images:

class ImageAugment:
    """
    Class that augments a batch of original images.

    Initialization attributes:
        :param operations: collection of image transformations to be implemented. It shoud follows
        from the same "data_augmentation" module.
        :type operations: tuple.
        :param augmentation_factor: proportion of augmented data with respect to the original batch. An
        augmentation_factor equals to 2.0 implies in an outcome batch with the double size of the original
        batch.
        :type augmentation_factor: float.

    Methods:
        "augment_data": method that creates new images from a provided batch of original images.
    """
    def __str__(self):
        params = ', '.join(
            [f'{k}={v}' for k, v in self.__dict__.items()]
            )
        return f'{self.__class__.__name__}({params})'

    def __repr__(self):
        return self.__str__()

    def __init__(self, operations: tuple, augmentation_factor: float = 2.0) -> None:
        self.operations = operations
        self.augmentation_factor = augmentation_factor
    
    def augment_data(self, images_batch: np.ndarray, labels: list) -> tuple:
        """
        Method that creates new images from a provided batch of original images.

        :param images_batch: collection of original images from which an augmented batch of data is created.
        :type images_batch: nd-array.
        :param labels: label of each provided original image.
        :type labels: list.

        :return: augmented batch of images and their respective labels.
        :rtype: tuple.
        """
        augmented_labels = deepcopy(labels)
        new_images, new_labels = [], []

        # Loop over provided images:
        for i in range(len(images_batch)):
            new_imgs, new_labels_ = self.__create_images(image=images_batch[i], label=labels[i])
            new_images.extend(new_imgs)
            new_labels.extend(new_labels_)
        
        augmented_imgs = np.vstack([images_batch, np.array(new_images)])
        augmented_labels.extend(new_labels)
        
        return augmented_imgs, augmented_labels
    
    def __create_images(self, image: np.ndarray, label: int) -> tuple:
        new_imgs, new_labels_ = [], []

        # Deterministically creating new images from an original image:
        for i in range(int(self.augmentation_factor) - 1):
            new_imgs.append(self.__transform(image=image))
            new_labels_.append(label)

        # Randomly creating new images from an original image:
        aug_prob = self.augmentation_factor - int(self.augmentation_factor)
        if np.random.choice([True, False], size=1, replace=False, p=[aug_prob, 1-aug_prob])[0]:
            new_imgs.append(self.__transform(image=image))
            new_labels_.append(label)

        return new_imgs, new_labels_

    def __transform(self, image: np.ndarray) -> np.ndarray:
        transf_img = image.copy()

        # Loop over transformations:
        for oper in self.operations:
            transf_img = oper.transform(transf_img)
        
        return transf_img

####################################################################################################################################
# Class that augments a batch of original images using Keras API:

class KerasImageAugment:
    """
    Class that augments a batch of original images using Keras API.

    Initialization attributes:
        :param oper_params: dictionary of parameters for transforming images using Keras API. Names of
        parameters follow from the kwargs of ImageDataGenerator class from Keras.
        :type oper_params: dictionary.
        :param augmentation_factor: proportion of augmented data with respect to the original batch. An
        augmentation_factor equals to 2.0 implies in an outcome batch with the double size of the original
        batch.
        :type augmentation_factor: float.

    Methods:
        "augment_data": method that creates new images from a provided batch of original images.
    """
    def __str__(self):
        params = ', '.join(
            [f'{k}={v}' for k, v in self.__dict__.items()]
            )
        return f'{self.__class__.__name__}({params})'

    def __repr__(self):
        return self.__str__()

    def __init__(self, oper_params: dict, augmentation_factor: float = 2.0) -> None:
        self.oper_params = oper_params
        self.augmentation_factor = augmentation_factor

    def augment_data(self, images_batch: np.ndarray, labels: list) -> tuple:
        """
        Method that creates new images from a provided batch of original images.

        :param images_batch: collection of original images from which an augmented batch of data is created.
        :type images_batch: nd-array.
        :param labels: label of each provided original image.
        :type labels: list.

        :return: augmented batch of images and their respective labels.
        :rtype: tuple.
        """
        augmented_imgs = images_batch.copy()
        augmented_labels = deepcopy(labels)

        # Deterministically creating new images from an original image:
        for i in range(int(self.augmentation_factor) - 1):
            new_images, new_labels = self.__deterministic_augment(images_batch=images_batch, labels=labels)
            augmented_imgs = np.vstack([augmented_imgs, np.array(new_images)])
            augmented_labels.extend(new_labels)
    
        # Randomly creating new images from an original image:
        if self.augmentation_factor - int(self.augmentation_factor) > 0:
            aug_prob = self.augmentation_factor - int(self.augmentation_factor)
            new_images, new_labels = self.__random_augment(images_batch=images_batch, labels=labels, aug_prob=aug_prob)
            augmented_imgs = np.vstack([augmented_imgs, np.array(new_images)])
            augmented_labels.extend(new_labels)

        return augmented_imgs, augmented_labels

    def __deterministic_augment(self, images_batch: np.ndarray, labels: list) -> tuple:
        # Data generator object:
        datagen = ImageDataGenerator(**self.oper_params)

        # Creating the iterator:
        iter = datagen.flow(x=images_batch, shuffle=False, batch_size=len(images_batch))

        # Applying transformations:
        new_batch = iter.next()

        new_images, new_labels = [], []

        # Converting pixels to unsigned integers for viewing:
        for i in range(len(new_batch)):
            new_images.append(new_batch[i].astype('uint8'))
            new_labels.append(labels[i])

        return new_images, new_labels

    def __random_augment(self, images_batch: np.ndarray, labels: list, aug_prob: float) -> tuple:
        random_idxs = list(np.random.choice(range(len(images_batch)), size=int(len(images_batch)*aug_prob), replace=False))
        random_imgs = images_batch[random_idxs]
        random_labels = [labels[i] for i in random_idxs]

        return self.__deterministic_augment(images_batch=random_imgs, labels=random_labels)
