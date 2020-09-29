"""The implementation of the Encryption and Decryption for the Elliptical Cryptography."""

# import matplotlib.pyplot as plt
import itertools
# import cv2
def snake(img, encryption="Encrypt"):
    """Snake function takes two arguments: csv image and "Encrypt"/"Decrypt"
    returns either image encrypted by Snake Addition or Decrypted by Snake addition. """
    dim_m = img.shape[0] # the dimensions are number of pixels, don't start with 0
    dim_n = img.shape[1]
    product = list(itertools.product(range(dim_m), range(dim_n)))
    product_reverse = list(reversed(product))
    if encryption == "Encrypt":
        print("Encrypting by snake")
        for i, j in product:
            if (i != 0 or j != 0):
                img[i, j] = int(img[i, j-1])+int(img[i, j])
            img[1, 1] = int(img[dim_m-1, dim_n-1])+int(img[1, 1])
    elif encryption == "Decrypt":
        print("Decrypting snake")
        for i, j in product_reverse:
            img[1, 1] = int(img[1, 1])-int(img[dim_m-1, dim_n-1])
            if (i != 0 or j != 0):
                img[i, j]=int(img[i, j])-int(img[i, j-1])
    return(img)











##################TEST FOR SNAKE FUNCTimgON############
# img=cv2.imread("Test.png",cv2.imgMREAD_GRAYSCALE)
# plt.imshow(img, cmap='gray')
# plt.show()
# img_snakeEncrypted=snake(img,"Encrypt")
# plt.imshow(img_snakeEncrypted, cmap='gray')
# plt.show()
# img_snakeDecrypted=snake(img_snakeEncrypted,"Decrypt")
# plt.imshow(img_snakeDecrypted, cmap='gray')
# plt.show()

