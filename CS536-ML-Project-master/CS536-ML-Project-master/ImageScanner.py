# import cv2
from PIL._imaging import display
from skimage import io, exposure
from skimage.transform import resize
import numpy as np
from patchify import patchify
from PIL import Image
# import matplotlib.pyplot as plt


def read_image(img_path):
    return io.imread(img_path)


def display_image(image):
    io.imshow(image)
    io.show()


def resize_image(image):
    return resize(image, (128, 128), anti_aliasing=True)


def create_patches(image, square_patch_size):
    np_img = np.asarray(image)
    return patchify(np_img, (square_patch_size, square_patch_size, 3), step=square_patch_size)


def create_horizontal_patch(patches, image_size, patch_size):
    output_patch = np.empty((image_size // patch_size, patch_size, image_size, 3))
    for row in range(patches.shape[0]):
        strip = patches[row]
        patch_hor_list = []
        for col in range(patches.shape[1]):
            patch_hor_list.append(strip[col, 0])
        output_patch[row] = np.hstack(patch_hor_list)
    return output_patch


def create_vertical_patch(patches, image_size, patch_size):
    output_patch = np.empty((image_size // patch_size, image_size, patch_size, 3))
    for col in range(patches.shape[1]):
        strip = patches[:, col]
        patch_vert_list = []
        for row in range(patches.shape[0]):
            patch_vert_list.append(strip[row, 0])
        output_patch[col] = np.vstack(patch_vert_list)
    return output_patch

#
def get_image(image_path, train=None):
    image_name, label_name, root_name = image_path
    if root_name == "mvtec":
        label_path = f"{root_name}/{label_name}"
        train_path = None
        if train is True:
            train_path = f"{label_path}/train"
        else:
            train_path = f"{label_path}/test"
        images_path = f"{train_path}/good"
    else:
        label_path = f"{root_name}/{label_name}"
        images_path = f"{label_path}/head_ct"

    return io.imread(f'{images_path}/{image_name}')

'''
image_path -> [image_name, label_name, root_name]
square_patches -> list of all patches
square_patch_size -> 16 as defined by the paper
patch_type -> X or Y string input
train -> boolean: True -> Training dataset else test

Returns: 
1. square patches
2. strips: 8x128x16x3 -> Thus transpose the horizontal ones to same shape
'''
def process_image(image_path, square_patch_size, patch_type, train):
    img = get_image(image_path, train)
    resized_img = resize_image(img)
    print(resized_img.shape)
    resized_img_size = resized_img.shape[0]
    square_patches = create_patches(resized_img, square_patch_size)
    strips = None
    if patch_type == 'X':
        strips = create_horizontal_patch(square_patches, resized_img_size, square_patch_size)
    elif patch_type == 'Y':
        strips = create_vertical_patch(square_patches, resized_img_size, square_patch_size)
    else:
        raise 'Incorrect Patch Type'

    strip_shape = (8, 128, 16, 3)
    if strips.shape != strip_shape:
        strips = np.transpose(strips, (0, 2, 1, 3))

    return [square_patches, strips]


# root_path = "mvtec"
# label_path = f"{root_path}/bottle"
# train_path = f"{label_path}/train"
# images_path = f"{train_path}/good"
# img = io.imread(f'{images_path}/000.png')
# # path = "mvtec/bottle/train/good/"
# # img_name = f'{path}/000.png'
# # img = read_image(path)
# # print(im)
# print('Image Shape          : ', img.shape)
# # display_image(img)
#
# resized_img = resize_image(img)
# print('Resized Image Shape  : ', resized_img.shape)
# # display_image(resized_img)
#
# patch_size = 16
# patches = create_patches(resized_img, patch_size)
# print('Patch Shape          : ', patches.shape)
#
# patch_img_arr=patches[3, 0, 0]
# print('Patch Image Shape    : ', patch_img_arr.shape)
# # patch_img=Image.fromarray(patch_img_arr)
# # display_image(patch_img_arr)
# # display(patch_img)
#
# horizontal_strips = create_horizontal_patch(patches, 128, patch_size)
#
# vertical_strips = create_vertical_patch(patches, 128, patch_size)
# print('Horizontal Image Patch Shape: ',np.transpose(horizontal_strips[1], (1, 0, 2)).shape)
# print('Vertical Image Patch Shape: ', vertical_strips.shape)
# print(horizontal_strips[1].reshape(vertical_strips[1].shape).shape)
# print(np.transpose(horizontal_strips, (0, 2, 1, 3)).shape)
#
# horizontal_strips = np.transpose(horizontal_strips, (0, 2, 1, 3))
#
# '''
# horizontal_strips[1].reshape(vertical_strips[1].shape) -> Extremely wrong, gives the wrong output image
# '''
# # print(np.transpose(horizontal_strips[1], (1, 0, 2)) == horizontal_strips[1].reshape(vertical_strips[1].shape))
# display_image(horizontal_strips[1])
# # display_image(np.transpose(horizontal_strips[1], (1, 0, 2)))
# # display_image(horizontal_strips[1].reshape(vertical_strips[1].shape))
#
# # display_image(vertical_strips[1])


# image_path -> [image_name, label_name, root_name]
image_path = ["000.png", "bottle", "mvtec"]
patch_size = 16
square_patches, strips = process_image(image_path, patch_size, 'Y', True)

print(square_patches.shape)
print(strips.shape)
display_image(square_patches[2, 0, 0])
display_image(strips[1])



