"""Methods representing various loss functions."""
import numpy as np
from keras import backend as K
from keras.layers import dot


def gram(matrix):
    """
    Return a gram matrix for the given input matrix.

    Args:
        matrix: the matrix to calculate the gram matrix of

    Returns: the gram matrix of `matrix`
    """
    return dot([matrix, matrix], axes=0)


def content_loss(content, combination):
    """
    Return the content loss between the content and combinations tensors.

    Args:
        content: the output of a layer for the content image
        combination: the output of a layer for the combination image

    Returns: the loss between `content` and `combination`
    """
    # squared euclidean distance, exactly how it is in the paper
    return 0.5 * K.sum(K.square(combination - content))


def style_loss(style, combination):
    """
    Return the style loss between the style and combinations tensors.

    Args:
        style: the output of a layer for the style image
        combination: the output of a layer for the combination image

    Returns: the loss between `style` and `combination`
    """
    # M_l is the width times the height of the current layer
    Ml = int(style.shape[0] * style.shape[1])
    # N_l is the number of distinct filters in the layer
    Nl = int(style.shape[2])

    # take the squared euclidean distance between the gram matrices of both
    # the style and combination image. multiply by the coefficient
    return K.sum(K.square(gram(style) - gram(combination))) / (4.0 * Nl**2 * Ml**2)


# def total_variation_loss(canvas):
#     h = canvas.height
#     w = canvas.width
#     a = K.square(canvas.output[:, :h-1, :w-1, :] - canvas.output[:, 1:, :w-1, :])
#     b = K.square(canvas.output[:, :h-1, :w-1, :] - canvas.output[:, :h-1, 1:, :])
#     return K.sum(K.pow(a + b, 1.25))


# explicitly export the public API
__all__ = [
    'content_loss',
    'style_loss'
]
