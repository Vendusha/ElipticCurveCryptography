"""The implementation of the Encryption and Decryption for the Elliptical Cryptography."""

import matplotlib.pyplot as plt
from numpy import *
import itertools
import cv2
import pdb
import numpy as np
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

def transform(img, num):
    rows, cols, ch = img.shape
    if (rows == cols):
        n = rows
        img2 = np.zeros([rows, cols, ch])

        for x in range(0, rows):
            for y in range(0, cols):

                img2[x][y] = img[(x+y)%n][(x+2*y)%n]

        cv2.imwrite("img/iteration" + str(num) + ".jpg", img2)
        return img2

    else:
        print("The image is not square.")

def arnold_map_modified(img, c, d, e, f, s, encryption="Encrypt"):
    """First and last argument: csv image and "Encrypt"/"Decrypt",
    in between are parameters for the modification.
    It returns image either encrypted or decrypted by modified arnold map matrix"""
    dim_m = img.shape[0] # the dimensions are number of pixels, don't start with 0
    dim_n = img.shape[1]
    if dim_m!=dim_n:
        print("Picture has wrong dimensions")
    new_img=img.copy()
    product = list(itertools.product(range(dim_m), range(dim_n)))
    if encryption == "Encrypt":
        print("Encrypting by modified Arnold map")
        for x, y in product:
            for index in range(s):
                new_img[x][y]=img[(x+c*y+e) % dim_m][(d*(x)+(c*d+1)*y+f) % dim_m]
    elif encryption == "Decrypt":
        print("Decrypting modified Arnold map")
        for x,y in product:
            for index in range(s):
                new_img[x][y]=img[((c*d+1)*(x-e)-c*(y-f)) % dim_m][(-d*(x-e)+y-f) % dim_m]
    return(new_img)
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
img=cv2.imread("Test.png",cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
plt.show()
img_Encrypted=arnold_map_modified(img,1,2,3,4,5,"Encrypt")
plt.imshow(img_Encrypted, cmap='gray')
plt.show()
img_Decrypted=arnold_map_modified(img_Encrypted,1,2,3,4,5,"Decrypt")
plt.imshow(img_Decrypted, cmap='gray')
plt.show()
