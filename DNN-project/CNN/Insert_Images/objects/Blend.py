# -*- coding: utf-8 -*-
"""
Created on Wed May 29 10:24:25 2019

@author: Usuario
"""



import numpy as np
import cv2
 
# Cargamos la imagen
original = cv2.imread("potato/12.jpg")
cv2.imshow("original", original)

# Convertimos a escala de grises
gris = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
 
# Aplicar suavizado Gaussiano
gauss = cv2.GaussianBlur(gris, (5,5), 0)
 
cv2.imshow("suavizado", gauss)

# Detectamos los bordes con Canny
canny = cv2.Canny(gauss, 50, 150)
 
cv2.imshow("canny", canny)

# Buscamos los contornos
(contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Mostramos el número de monedas por consola
print("He encontrado {} objetos".format(len(contornos)))

cv2.drawContours(original,contornos,-1,(0,0,255), 2)
cv2.imshow("contornos", original)

cv2.waitKey(0)


#from PIL import Image
#import matplotlib.pyplot as plt
#
## Function to change the image size
#def changeImageSize(maxWidth, 
#                    maxHeight, 
#                    image):
#    
#    widthRatio  = maxWidth/image.size[0]
#    heightRatio = maxHeight/image.size[1]
#
#    newWidth    = int(widthRatio*image.size[0])
#    newHeight   = int(heightRatio*image.size[1])
#
#    newImage    = image.resize((newWidth, newHeight))
#    return newImage
#
#im1 = Image.open("01.jpg")
#im2 = Image.open("potato/1.png")
#
## Make the images of uniform size
#image3 = changeImageSize(800, 500, im1)
#image4 = changeImageSize(800, 500, im2)
#
## Make sure images got an alpha channel
#image5 = image3.convert("RGBA")
#image6 = image4.convert("RGBA")
#
## Display the images
#image5.show()
#image6.show()
#
## alpha-blend the images with varying values of alpha
#alphaBlended1 = Image.blend(image5, image6, alpha=0.5)
##alphaBlended2 = Image.blend(image5, image6, alpha=0.4)
#
## Display the alpha-blended images
#alphaBlended1.show()
##alphaBlended2.show()
#alphaBlended1.save("alphaBlended1.png")
##blended = Image.blend(im1, im2, alpha=0.5)
##blended.save("blended.png")