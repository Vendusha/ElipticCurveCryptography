"""The implementation of the Encryption and Decryption for the Elliptical Cryptography."""

import matplotlib.pyplot as plt
from numpy import *
import itertools
import cv2
import os
import numpy as np
from Crypto.PublicKey import ECC
from Crypto.Random import random
from Crypto.Util import number
def merge_image(img_array,m,n):
    img_merge = np.zeros((m*256,n*256),np.uint8)
    product = list(itertools.product(range(256), range(256)))
    i=0
    j=0
    img=img_array[0]
    for index, img in enumerate(img_array):
        img_merge[i*256:(i+1)*256,j*256:(j+1)*256]=img
        j+=1
        if (index+1)%(m+1)==0 and index>1:
            i+=1
            j=0
    return(img_merge)
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
        # plt.imshow(img_new,cmap='gray')
        # plt.show()
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
    return(array_of_images,no_of_images_horizontal,no_of_images_vertical)
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

def key_generation_Alice():
    """ Generate ECC key """
    key=ECC.generate(curve='P-256')
    private_key_file = open('AlicePrivateKey.pem', 'wt')
    private_key_file.write(key.export_key(format='PEM'))
    private_key_file.close()
    public_key = key.public_key()
    return(public_key.export_key(format="PEM"))
    # return
def key_matrix_generator(K_0,L,array_of_images):
    """Key Matrix Generator"""
    ##########Key Matrix Generation according to the article#############
    Keyseq=""
    i=0
    K=K_0
    K_M=np.empty((len(array_of_images)+1,256,256),dtype=int)
    K_M_modified=np.zeros((len(array_of_images)+1,256,256),dtype=int)
    K_M_0=np.zeros((256,256),dtype=int)
    while len(Keyseq)<2**19:
        i+=1
        K+=K
        Keyseq=Keyseq+bin(int(K.x)^int(L.x))[2:]+bin((int(K.y)^int(L.y)))[2:]
    for i in range(2**8):
        for j in range(2**8):
            ptr=2**11*(i)+2**3*(j)
            element=Keyseq[ptr:ptr+8]
            K_M_0[i,j]=int(element, 2)
    K_M[0]=K_M_0
    a=np.zeros((5),dtype=int)
    for index, element in enumerate(array_of_images):
        # plt.imshow(element,cmap='gray')
        # plt.show()
        i=index+1
        for j in range(5):
            a[j]=np.int((int(K_M_0[1,j])^int(K_M[i-1,j,255])))
        # print(K_M[i]+K_M_0,a[0],a[2],a[2],a[3],a[4])
        K_M_modified[i]=arnold_map_modified(K_M[i-1]+K_M_0, a[0], a[1], a[2], a[3], a[4])
        for j in range(256):
            for k in range (256):
                K_M[i][j][k]=(K_M_modified[i][j][k])^(K_M_0[j][k])
    return K_M

def key_matrix_generator_Bob(public_key,array_of_images):
    """Key Matrix Generator"""
    key=ECC.import_key(public_key) #importing private key from Alice
    Q=key.pointQ
    P=ECC.EccPoint(x=key._curve.Gx,y=key._curve.Gy, curve='P-256')
    n=32
    k_b = (random.randrange(2**(n-1)+1, 2**n - 1) )
    k_c = (random.randrange(2**(n-1)+1, 2**n - 1))
    ##note call the proper definition, 
    R=k_b*P
    S=k_c*P
    K_0 = k_b*Q
    L = k_c*Q
    K_M=key_matrix_generator(K_0,L,array_of_images)
      # np.savetxt("Matrix.txt",(K_M))
    return K_M,R,S

def Encryption(K_M,M,m,n):
    """Encryption module"""
    ####Declaration#####
    C=np.zeros((len(M),256,256),dtype=int)
    M_modified=np.zeros((len(M),256,256),dtype=int)
    ###Addition####
    for i in range(len(M)):
        M_modified[i]=M[i]+K_M[i]
    ###SnakeAddition########
    img=merge_image(M_modified,m,n)
    img=snake(img,"Encrypt")
    (M,m,n)=preprocessing(img)
    ####Modified Arnold map ####
    K_M_0=K_M[0]
    b=np.zeros((5),dtype=int)
    for i, element in enumerate(M):
        # plt.imshow(element,cmap='gray')
        # plt.show()
        for j in range(5):
            b[j]=np.int((int(K_M_0[j,255])^int(K_M[i,j,0])))
        # print(M[i],b[0],b[2],b[2],b[3],b[4])
        C[i]=arnold_map_modified(M[i], b[0], b[1], b[2], b[3], b[4])
    ######Snake Addition######
    img=merge_image(C,m,n)
    img=snake(img,"Encrypt")
    (C,m,n)=preprocessing(img)
    ########XOR Operation######
    for i, element in enumerate(M):
        for j in range(256):
            for k in range (256):
                C[i][j][k]=(C[i][j][k])^(K_M[i][j][k])
    img=merge_image(C,m,n)
    return(img)

def Decryption(R,S,C):
    f = open('AlicePrivateKey.pem','rt')
    key = ECC.import_key(f.read())
    k_a=key.d
    K_0=k_a*R
    L=k_a*S
    (C,m,n)=preprocessing(C)
    K_M=key_matrix_generator(K_0,L,C)
    M=np.zeros((len(C),256,256),dtype=int)
    for i, element in enumerate(M):
        for j in range(256):
            for k in range (256):
                C[i][j][k]=(C[i][j][k])^(K_M[i][j][k])
    img=merge_image(C,m,n)
    img=snake(img,"Decrypt")
    (C,m,n)=preprocessing(img)
    K_M_0=K_M[0]
    b=np.zeros((5),dtype=int)
    for i, element in enumerate(C):
        for j in range(5):
            b[j]=np.int((int(K_M_0[j,255])^int(K_M[i,j,0])))
        # print(M[i],b[0],b[2],b[2],b[3],b[4])
        M[i]=arnold_map_modified(C[i], b[0], b[1], b[2], b[3], b[4],"Decrypt")
    img=merge_image(M,m,n)
    img=snake(img,"Decrypt")
    (M,m,n)=preprocessing(img)
    for i in range(len(M)):
        M[i]=M[i]-K_M[i]
    img=merge_image(M,m,n)
    return img

#####################################################################################
#########################ENCRYPTION##################################################
#####################################################################################
public_key=key_generation_Alice()
print("Public key Alice generated.")

# img=cv2.imread("Test.png",cv2.IMREAD_GRAYSCALE)
# img=cv2.imread("test_image.jpg",cv2.IMREAD_GRAYSCALE)
img=cv2.imread("test_image_simple.jpg",cv2.IMREAD_GRAYSCALE)
plt.imshow(img,cmap="gray")
plt.show()
(array_of_images,m,n)=preprocessing(img)
for z in (array_of_images):
        plt.imshow(z,cmap='gray')
        plt.show()
img=merge_image(array_of_images,m,n)
plt.imshow(img,cmap="gray")
plt.show()

print("Preprocessing done.")
K_M,R,S=key_matrix_generator_Bob(public_key, array_of_images)
print("Key matrix generated.")
C=Encryption(K_M, array_of_images,m,n)
plt.imshow(C,cmap="gray")
plt.show()
print("Picture encrypted")
#####Saving encrypted picture##########################################
f = open('Encrypted.txt','w') 
f.write(str(R.x))
f.write("\n")
f.write(str(R.y))
f.write("\n")
f.write(str(S.x))
f.write("\n")
f.write(str(S.y))
f.write("\n")
f.close()
print(C)
np.savetxt("C.txt",C)

#####################################################################################
#########################DECRYPTION##################################################
#####################################################################################

###################Loading encrypted picture########################################
f = open('Encrypted.txt','r')
R_x=f.readline()
R_y=f.readline()
S_x=f.readline()
S_y=f.readline()
f.close()
C= np.loadtxt("C.txt")
R=ECC.EccPoint(x=R_x,y=R_y)
S=ECC.EccPoint(x=S_x,y=S_y)
plt.imshow(C,cmap='gray')
plt.show()
D=Decryption(R,S,C)
print("Picture decrypted")
plt.imshow(D,cmap='gray')
plt.show()


#######################TEST FOR MERGING IMAGES############
# img=cv2.imread("Test.png",cv2.IMREAD_GRAYSCALE)
# # img=cv2.imread("test_image.jpg",cv2.IMREAD_GRAYSCALE)
# plt.imshow(img,cmap='gray')
# plt.show()
# (array_of_images,m,n)=preprocessing(img)
# merged_image=merge_image(array_of_images,m,n)
# plt.imshow(merged_image,cmap='gray')
# plt.show()

##################TEST FOR PREPROCESSING############
# img=cv2.imread("test_image.jpg",cv2.IMREAD_GRAYSCALE)
# plt.imshow(img,cmap='gray')
# plt.show()
# (array_of_images,m,n)=preprocessing(img)
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
