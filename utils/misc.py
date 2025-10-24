import numpy as np


def denormalize_image_for_display(tensor_image):
    """Reverses ImageNet normalization for display.

    Expects a CHW numpy array or tensor-like that can be transposed.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    denormalized_img = (
        ((tensor_image.transpose(1, 2, 0) * std) + mean) * 255.0
    ).astype(np.uint8)
    return denormalized_img
