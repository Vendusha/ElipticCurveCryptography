"""The implementation of the Encryption and Decryption for the Elliptical Cryptography."""

import matplotlib.pyplot as plt
from numpy import *
import itertools
import cv2
import pdb
# from PIL import Image
def snake(img, encryption="Encrypt"):
    """Snake function takes two arguments: csv image and "Encrypt"/"Decrypt"
    returns either image encrypted by Snake Addition or Decrypted by Snake addition. """
    dim_m = img.shape[0] # the dimensions are number of pixels, don't start with 0
    dim_n = img.shape[1]
    product = list(itertools.product(range(dim_m), range(dim_n)))
    product_reverse = list(reversed(product))
    if encryption == "Encrypt":
        print("Encrypting by snake")
        previous=int(img[0,0])
        for i, j in product:
            if (i != 0 and j != 0):
                img[i, j] += previous
                previous=int(img[i,j])
        img[0,0] += int(img[dim_m-1, dim_n-1])
    elif encryption == "Decrypt":
        print("Decrypting snake")
        img[0, 0] -= int(img[dim_m-1, dim_n-1])
        previous=(dim_m-1,dim_n-1)
        for i, j in (product_reverse):
            if (i != 0 and j != 0):
                img[previous] -= int(img[i,j])
                previous=(i,j)
    return img

def arnold_map_modified(img, var_c, var_d, var_e, var_f, var_s, encryption="Encrypt"):
    """First two arguments: csv image and "Encrypt"/"Decrypt",
    after that the arguments are parameters for the modification.
    It returns image either encrypted or decrypted by modified arnold map matrix"""
    dim_m = img.shape[0] # the dimensions are number of pixels, don't start with 0
    dim_n = img.shape[1]
    new_img=img
    product = list(itertools.product(range(dim_m), range(dim_n)))
    if encryption == "Encrypt":
        print("Encrypting by modified Arnold map")
        for i,j in product:
            for index in range(s):
                x_arnold=i+j
            new_img(x,y)
    # im = array(Image.open("cat.jpg"))
    # N = im.shape[0]

    # # create x and y components of Arnold's cat mapping
    x,y = meshgrid(range(N),range(N))
    print(x)
    print(y)
    pdb.set_trace()
    # xmap = (2*x+y) % N
    # ymap = (x+y) % N

    # for i in range(N+1):
    #     result = Image.fromarray(im)
    #     result.save("cat_%03d.png" % i)
    #     im = im[xmap,ymap]

    # x_grid, y_grid = meshgrid(range(dim_m),range(dim_n))
    # if dim_m != dim_n:
    #     print("The pixels are not squared, cannot apply ArnoldMap")
    # if encryption == "Encrypt":
    #     for _ in range(var_s):
    #         x_map = (x_grid+var_c*y_grid)+ var_e % dim_m
    #         y_map = (var_d*x_grid+var_c*var_d*y_grid) +var_f % dim_m
    #         img=img(dstack((x_map,y_map)))
    #         plt.imshow(img, cmap="gray")
    #         plt.show()
    # elif encryption == "Decrypt":
        # x_map = ()
        # y_map = ()
    return(img)
# img=cv2.imread("Test.png",cv2.IMREAD_GRAYSCALE)
# arnold_map_modified(img,1,2,3,4,5)
##################TEST FOR SNAKE FUNCTION############
# img=cv2.imread("Test.png",cv2.IMREAD_GRAYSCALE)
# plt.imshow(img, cmap='gray')
# plt.show()
# img_snakeEncrypted=snake(img,"Encrypt")
# plt.imshow(img_snakeEncrypted, cmap='gray')
# plt.show()
# img_snakeDecrypted=snake(img_snakeEncrypted,"Decrypt")
# plt.imshow(img_snakeDecrypted, cmap='gray')
# plt.show()
##################TEST FOR THE ARNOLD MAP############
# img=cv2.imread("Test.png",cv2.IMREAD_GRAYSCALE)
# plt.imshow(img, cmap='gray')
# plt.show()
# img_Encrypted=arnold_map_modified(img,"Encrypt")
# plt.imshow(img_Encrypted, cmap='gray')
# plt.show()
# img_Decrypted=arnold_map_modified(img_Encrypted,"Decrypt")
# plt.imshow(img_Decrypted, cmap='gray')
# plt.show()
