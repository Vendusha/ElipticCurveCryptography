"""The implementation of the Encryption and Decryption for the Elliptical Cryptography."""

import matplotlib.pyplot as plt
from numpy import *
import itertools
import cv2
import os
import pdb
import numpy as np
from Crypto.PublicKey import ECC
def preprocessing(img):
    # """Loads an image and splits it in the t 256x256 sub-images."""
    dim_m = img.shape[0]
    dim_n= img.shape[1]
    array_of_images=[]
    if dim_n %256 !=0 or dim_m %256 !=0:
        if dim_m % 256 != 0:
            for i in range(256):
                if (dim_m+i)%256 ==0:
                    dim_m_new=dim_m+i
                    break
        if dim_n % 256 !=0:
            for i in range(256):
                if (dim_n+i)%256 ==0:
                    dim_n_new=dim_n+i
                    break
        img_new = np.zeros((dim_m_new,dim_n_new),np.uint8)
        product = list(itertools.product(range(dim_m_new), range(dim_n_new)))
        for i, j in product:
            if i<dim_m and j<dim_n:
                img_new[i][j]=img[i][j]
            else:
                img_new[i][j]=0 # filling the extra dimensions by random bits
        plt.imshow(img_new,cmap='gray')
        plt.show()
    else:
        img_new=img
    dim_m = img_new.shape[0]
    dim_n= img_new.shape[1]
    no_of_images_horizontal=int(dim_m/256)
    no_of_images_vertical=int(dim_n/256)
    product = list(itertools.product(range(256), range(256)))
    for i in range(no_of_images_horizontal):
        for j in range(no_of_images_vertical):
            image=np.zeros((256,256),np.uint8)
            for k,l in product:
                image[k][l]=img_new[256*i+k][256*j+l]
            array_of_images.append(image)
    return(array_of_images)
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
def key_generation():
    """ Helper function to retrieve public key """
    key=ECC.generate(curve='P-256')
    private_key_file = open('AlicePrivateKey.pem', 'wt')
    private_key_file.write(key.export_key(format='PEM'))
    private_key_file.close()
    public_key = key.public_key()
    return public_key.export_key(format="PEM")

##################TEST FOR PREPROCESSING############
# img=cv2.imread("test_image.jpg",cv2.IMREAD_GRAYSCALE)
# plt.imshow(img,cmap='gray')
# plt.show()
# array_of_images=preprocessing(img)
# for z in (array_of_images):
#         plt.imshow(z,cmap='gray')
#         plt.show()

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
# img_Encrypted=arnold_map_modified(img,1,2,3,4,5,"Encrypt")
# plt.imshow(img_Encrypted, cmap='gray')
# plt.show()
# img_Decrypted=arnold_map_modified(img_Encrypted,1,2,3,4,5,"Decrypt")
# plt.imshow(img_Decrypted, cmap='gray')
# plt.show()
