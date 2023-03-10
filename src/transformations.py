####################################################################################################################################
####################################################################################################################################
#############################################################LIBRARIES##############################################################

import numpy as np

####################################################################################################################################
####################################################################################################################################
########################################################FUNCTIONS AND CLASSES#######################################################

####################################################################################################################################
# Function that changes the brightness (average color intensity across all color channels) of a given image:

def brightness_filter(image: np.ndarray, delta: float) -> np.ndarray:
    """
    Function that changes the brightness (average color intensity across all color channels) of a given image.

    :param image: matrices with RGB color intensities.
    :type image: ndarray whose dimensions are the height, the width and the number of color channels (i.e., 3).
    :param delta: proportion of change in brightness to be imposed on the image. delta = 1 means doubling the
    overall brightness, while negative values imply a reduction.
    in brightness.
    :type delta: float.

    :return: transformed image with brightness added by delta.
    :rtype: ndarray.
    """
    image = image.copy()
    height, width, _ = image.shape

    # Loop over horizontal positions:
    for w in range(width):
      # Loop over vertical positions:
        for h in range(height):
            r, g, b = image[h, w, 0], image[h, w, 1], image[h, w, 2]

            # Adding (or subtracting) brightness to the image:
            r_ = min(255, max(0, int(r*(1+delta))))
            g_ = min(255, max(0, int(g*(1+delta))))
            b_ = min(255, max(0, int(b*(1+delta))))

            image[h, w, 0] = int(r_)
            image[h, w, 1] = int(g_)
            image[h, w, 2] = int(b_)

    return image

####################################################################################################################################
# Function that changes the contrast of a given image:

def contrast_filter(image: np.ndarray, beta: int) -> np.ndarray:
    """
    Function that changes the contrast of a given image.

    :param image: matrices with RGB color intensities.
    :type image: ndarray whose dimensions are the height, the width and the number of color channels (i.e., 3).
    :param beta: variation on contrast to be imposed on the image.
    :type beta: integer.

    :return: transformed image with contrast added by beta.
    :rtype: ndarray.
    """
    image = image.copy()
    height, width, _ = image.shape

    # Average brightness:
    mu = np.mean(image, axis=2)
    mu_mean = mu.mean()

    # Calculating contrast factor:
    if beta == 255:
        alpha = np.infty
    else:
        alpha = (255+beta)/(255-beta)

    # Loop over horizontal positions:
    for w in range(width):
      # Loop over vertical positions:
        for h in range(height):
            r, g, b = image[h, w, 0], image[h, w, 1], image[h, w, 2]

            # Adding (or subtracting) contrast to the image:
            r_ = min(255, max(0, alpha*(r - mu_mean) + mu_mean))
            g_ = min(255, max(0, alpha*(g - mu_mean) + mu_mean))
            b_ = min(255, max(0, alpha*(b - mu_mean) + mu_mean))

            image[h, w, 0] = int(r_)
            image[h, w, 1] = int(g_)
            image[h, w, 2] = int(b_)

    return image

####################################################################################################################################
# Function that changes the saturation of a given image:

def saturation_filter(image: np.ndarray, beta: int) -> np.ndarray:
    """
    Function that changes the saturation of a given image.

    :param image: matrices with RGB color intensities.
    :type image: ndarray whose dimensions are the height, the width and the number of color channels (i.e., 3).
    :param beta: variation on saturation to be imposed on the image.
    :type beta: integer.

    :return: transformed image with saturation added by beta.
    :rtype: ndarray.
    """
    image = image.copy()
    height, width, _ = image.shape

    # Calculating contrast factor:
    if beta == 255:
        alpha = np.infty
    else:
        alpha = (255+beta)/(255-beta)

    # Loop over horizontal positions:
    for w in range(width):
      # Loop over vertical positions:
        for h in range(height):
            r, g, b = image[h, w, 0], image[h, w, 1], image[h, w, 2]

            # Brightness:
            mu = (r + g + b)/3

            # Adding (or subtracting) saturation to the image:
            r_ = min(255, max(0, alpha*(r - mu) + mu))
            g_ = min(255, max(0, alpha*(g - mu) + mu))
            b_ = min(255, max(0, alpha*(b - mu) + mu))

            image[h, w, 0] = int(r_)
            image[h, w, 1] = int(g_)
            image[h, w, 2] = int(b_)

    return image

####################################################################################################################################
# Class that scales a batch of images:

class ScaleImage:
    """
    Class that scales a batch of images considering both normalization of pixels
    (divides each pixel by a rescale factor, usually 255) and centering of pixels.

    Initialization attributes:
        :param rescale: indicates whether to divide pixel values by some rescale factor.
        :type rescale: boolean.
        :param centering: indicates whether to subtract pixel values by averages of pixels.
        :type centering: boolean.

    Methods:
        "transform": method that scales the pixels of provided images.
    """
    def __str__(self):
        params = ', '.join(
            [f'{k}={v}' for k, v in self.__dict__.items()]
            )
        return f'{self.__class__.__name__}({params})'

    def __repr__(self):
        return self.__str__()

    def __init__(self, rescale: bool = True, centering: bool = False, scale: int = 255) -> None:
        self.rescale = rescale
        self.centering = centering
        self._scale = scale

    def transform(self, images_batch: np.ndarray) -> np.ndarray:
        """
        Method that scales the pixels of provided images.

        :param images_batch: collection of original images from which an augmented batch of data is created.
        :type images_batch: nd-array.

        :return: batch of scaled images.
        :rtype: nd-array.
        """
        scaled_imgs = []

        # Loop over provided images:
        for i in range(len(images_batch)):
            scaled_imgs.append(self.__transform(image=images_batch[i]))

        return np.array(scaled_imgs)

    def __transform(self, image: np.ndarray) -> np.ndarray:
        transf_img = image.copy()

        if self.centering:
            transf_img = self.__centering_pixels(image=transf_img)
        
        if self.rescale:
            transf_img = self.__rescale_image(image=transf_img, scale=self._scale)
        
        return transf_img
    
    def __centering_pixels(self, image: np.ndarray) -> np.ndarray:
        pass
      
    def __rescale_image(self, image: np.ndarray, scale: int = 255) -> np.ndarray:
        return image/scale
