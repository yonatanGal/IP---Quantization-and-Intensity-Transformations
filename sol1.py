import numpy as np
from imageio import imread
from skimage.color import *
import matplotlib.pyplot as plt

BINS = 256
NORMALIZE = 255
R_CHANNEL = 0
G_CHANNEL = 1
B_CHANNEL = 2
RGB_SHAPE = 3
Y_CHANNEL = 0
I_CHANNEL = 1
Q_CHANNEL = 2
GRAY = 1
RGB = 2
ERROR_MSG = "representation can only be 1 or two. please try again"
YIQ_MAT = np.array([[0.299,  0.587,  0.114],
                    [0.596, -0.275, -0.321],
                    [0.212, -0.523, 0.311]])


def read_image(filename, representation):
    """
    reads a given image into a representation type
    :param filename: a picture saved on the disk
    :param representation: can be 1 - grayscale, 2 - RGB
    :return: a matrix represents the input image, in RGB or grayscale.
    """
    image = imread(filename)
    np_image = np.array(image)
    if not isinstance(image, np.float64):
        np_image = np_image.astype(np.float64)
        np_image /= NORMALIZE
    if representation == GRAY:
        return rgb2gray(np_image)
    elif representation == RGB:
        return np_image
    else:
        return ERROR_MSG


def imdisplay(filename, representation):
    """
    displays an image
    :param filename: filename: a picture saved on the disk
    :param representation: representation: can be 1 - grayscale, 2 - RGB
    :return: popping up a new windwo and displays the picture according to the
             filename and representation given.
    """
    image = read_image(filename, representation)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()


def rgb2yiq(imRGB):
    """
    converts a rgb image into a yiq image
    :param imRGB: an image of RGB type
    :return: the image, converted to YIQ
    """
    yiq_im = np.empty(imRGB.shape)
    r_channel = imRGB[:, :, R_CHANNEL]
    g_channel = imRGB[:, :, G_CHANNEL]
    b_channel = imRGB[:, :, B_CHANNEL]
    for i in range(3):
        yiq_im[:, :, i] = YIQ_MAT[i][R_CHANNEL] * r_channel +\
                          YIQ_MAT[i][G_CHANNEL] * g_channel +\
                          YIQ_MAT[i][B_CHANNEL] * b_channel
    return yiq_im


def yiq2rgb(imYIQ):
    """
    converts a yiq image into a rgb image
    :param imYIQ: an image of YIQ type
    :return: the image, converted to RGB
    """
    YIQ_MAT_INV = np.linalg.inv(YIQ_MAT)
    rgb_im = np.empty(imYIQ.shape)
    y_channel = imYIQ[:, :, Y_CHANNEL]
    i_channel = imYIQ[:, :, I_CHANNEL]
    q_channel = imYIQ[:, :, Q_CHANNEL]
    for i in range(3):
        rgb_im[:, :, i] = YIQ_MAT_INV[i][Y_CHANNEL] * y_channel +\
                          YIQ_MAT_INV[i][I_CHANNEL] * i_channel + \
                          YIQ_MAT_INV[i][Q_CHANNEL] * q_channel
    return rgb_im


def histogram_equalize(im_orig):
    """
    peeforms histogram equalization
    :param im_orig: the image that we want to equalize
    :return: the image, equalized
    """
    if len(im_orig.shape) == RGB_SHAPE:
        yiq_im = rgb2yiq(im_orig)
        y_channel = yiq_im[:, :, Y_CHANNEL]
        y_channel *= NORMALIZE
        [eq_y_channel, hist_orig, hist_eq] = equalize(y_channel)
        yiq_im[:, :, Y_CHANNEL] = eq_y_channel
        im_eq = yiq2rgb(yiq_im)
        im_eq = np.clip(im_eq, 0, 1)
        return [im_eq, hist_orig, hist_eq]
    else:
        im_orig *= NORMALIZE
        im_eq, hist_orig, hist_eq = equalize(im_orig)
        im_eq = np.clip(im_eq, 0, 1)
        return [im_eq, hist_orig, hist_eq]


def equalize(im_orig):
    """
    this function makes the original histogram, the equalized histogram, and the
    equalized image.
    """
    hist_orig, bins = np.histogram(im_orig, bins=np.arange(BINS+1))
    his_com = np.cumsum(hist_orig)
    m = np.nonzero(his_com)[0]
    sm = his_com[m[0]]
    sk = his_com[-1]
    look_up_table = np.array(((his_com[np.arange(BINS)] - sm) * NORMALIZE) /
                             (sk - sm))
    im_eq_unnormalized = look_up_table[im_orig.astype(np.uint8)]
    hist_eq, bins1 = np.histogram(im_eq_unnormalized, bins=np.arange(BINS+1))
    im_eq = (im_eq_unnormalized / NORMALIZE).astype(np.float64)
    return [im_eq, hist_orig, hist_eq]


def quantize(im_orig, n_quant, n_iter):
    """
    :param im_orig: the original image that we want to quantize
    :param n_quant: number of segments that we want
    :param n_iter: maximum number of iterations that we do the procedure
    :return: the quantize image and the errors as an numpy array
    """
    error = []
    if len(im_orig.shape) == RGB_SHAPE:
        yiq_im = rgb2yiq(im_orig)
        y_channel = yiq_im[:, :, Y_CHANNEL]
        y_channel *= NORMALIZE
        y_channel_new, error = quantize_helper(error, y_channel, n_iter,n_quant)
        yiq_im[:, :, Y_CHANNEL] = y_channel_new
        im_quant = yiq2rgb(yiq_im)
        return im_quant, error
    else:
        im_orig = im_orig * NORMALIZE
        return quantize_helper(error, im_orig, n_iter, n_quant)


def quantize_helper(error, im_orig, n_iter, n_quant):
    """
    helper function for the quantize process
    :param error: the errors array
    :param im_orig: the original image
    :param n_iter: maximum number of iterations that we do the procedure
    :param n_quant: number of segments that we want
    :return: the quantize image and the errors array
    """
    hist, bins = np.histogram(im_orig, bins=np.arange(BINS+1))
    his_com = np.cumsum(hist)
    z_array = initialize_z(n_quant, his_com)
    q_array = initialize_q(n_quant, z_array, hist)
    counter = 0
    while counter < n_iter:
        last_z_array = z_array.copy()
        z_array = rearrange_z(z_array, q_array)
        if np.array_equal(last_z_array, z_array):
            break
        q_array = rearrange_q(z_array, q_array, hist)
        err = compute_error(z_array, q_array, hist)
        error.append(err)
        counter += 1
    if len(error) == 0:
        error.append(compute_error(z_array, q_array, hist))
    im_quant = quant_the_image(im_orig, n_quant, q_array, z_array)
    return im_quant, np.array(error)


def quant_the_image(im_orig, n_quant, q_array, z_array):
    """
    creates the look up table and performs it on the original image
    :param im_orig: the original image
    :param n_quant: number of segments that we want
    :param q_array: the q array
    :param z_array: the z array
    :return: the quantize image
    """
    look_up_table = np.empty(NORMALIZE + 1)
    z_array = np.round(z_array)
    q_array = np.round(q_array)
    for i in range(n_quant):
        look_up_table.put(range(z_array[i], z_array[i + 1]+1), q_array[i])
    im_qu_unnormalized = look_up_table[im_orig.astype(np.uint8)]
    im_quant = (im_qu_unnormalized / NORMALIZE).astype(np.float64)
    return im_quant


def initialize_z(n_quant, his_com):
    """
    initializes the first array of z's
    :param n_quant: number of segments that we want
    :param his_com: the cumulative histogram of the original image
    :return: the updated z array
    """
    z_array = np.arange(n_quant + 1)
    pixel_per_quant = int(his_com[-1] / n_quant)
    temp = pixel_per_quant
    z_array[0] = 0
    for i in range(n_quant):
        temp_array = np.where(his_com >= temp)
        z_array[i+1] = temp_array[0][0]
        temp += pixel_per_quant
    if z_array[-1] != NORMALIZE:
        z_array[-1] = NORMALIZE
    return z_array


def initialize_q(n_quant, z_array, hist):
    """
    initializes the first array of q's
    :param n_quant: number of segments that we want
    :param z_array: the z's array
    :param hist: the histogram of the original image
    :return: the updates q array
    """
    q_array = np.arange(n_quant)
    for k in range(len(q_array)):
        q_array[k] = np.round(np.average(range(z_array[k], z_array[k + 1]),
                                weights=hist[z_array[k]:z_array[k + 1]]))
    return q_array


def compute_error(z_array, q_array, hist):
    """
    computes the error each iteration
    :param z_array: the array of the z's
    :param q_array: the array of the q's
    :param hist: the histogram of the original image
    :return: the current error
    """
    err = 0
    for i in range(len(z_array) - 1):
        temp = np.dot(np.square((q_array[i])-
                                np.arange(z_array[i], z_array[i+1])),
                      hist[z_array[i]:z_array[i+1]])
        err += temp
    return err


def rearrange_q(z_array, q_array, hist):
    """
    computes the current q array
    :param z_array: the array of the z's
    :param q_array: the array of the q's
    :param hist: the histogram of the original image
    :return: the updated q array
    """
    for k in range(len(q_array)):
        q_array[k] = np.round(np.average(range(z_array[k], z_array[k + 1]),
                            weights=hist[z_array[k]:z_array[k + 1]]))
    return q_array


def rearrange_z(z_array, q_array):
    """
    computes the current z array
    :param z_array: the array of the z's
    :param q_array: the array of the q's
    :return: the updated z array
    """
    for i in range(1, z_array.size-1):
        z_array[i] = np.ceil((q_array[i - 1] + q_array[i]) / 2.0)
    return z_array