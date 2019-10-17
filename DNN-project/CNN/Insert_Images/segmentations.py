# -*- coding: utf-8 -*-
"""
Script to view how the program segmentates the objects before sticking them to a background. the segmentation is applied 
automatically to erase the white background.
To visualize the segmentation, stick an object on the black background.In case the segmentation leaves white areas it is 
convenient to paint such areas with an editor like GIMP to a greyish color so the segmentation could be done correctly.

"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt



dir_img='BG/1.jpg'
mask_dir='objects/potato/5' #No poner la extension

#TODO: Se podria implementar que cuando pegas la imagen no lo hagas tal cual sino que hagas un blending entre la imagen
      # de fondo (en este caso negra) y la imagen del objeto
       
#Para buscar la extension de la imagen automaticamente:
extensiones=['.jpg','.png','.jpeg']
for ext_i in extensiones:
    mask_dir2=mask_dir+ext_i
    if(os.path.exists(mask_dir2)):
        mask_dir=mask_dir2
        break
    
#Leemos las imagenes:
image_BG=cv2.imread(dir_img)
image=cv2.imread(mask_dir)


def genera_mascara(image, threhold_area=0.00, muestra_resultado=False):
    """
    Funcion que dada una imagen de entrada con un objeto sobre un fondo blanco,
    te devuelve la imagen recortada donde esta el objeto y la mascara binaria del objeto de la imagen
    
    ----------------------
    Parametros de entrada:
    ----------------------
    image: imagen de entrada con valores de pixeles de 0 a 255
    threshold_area: valor umbral en % para rellenar huecos en la mascara que sean menores que un determinado area (el % indica el porcentaje de area respecto del area del objeto que se quiere segmentar por debajo del cual se rellenan los huecos)
                    
    muestra_resultado: booleano para indicar si se quiere representar el objeto ya enmascarado
    
    ----------------------
    Parametros de salida:
    ----------------------   
    image_crop: imagen del objeto original, recortada con la Bounding box del objeto
    mask_crop: mascara binaria correspondiente del objeto en la imagen recortada
    
    """

    #Transformacion de imagen RGB a grey
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #Aplicacion de filtro gaussiano para eliminar ruido gausiano
    gray = cv2.GaussianBlur(gray, (7, 7), 3)
    
    #Aplicacion de filtro de mediana para eliminar outliers
#    gray = cv2.medianBlur(gray, 3) # 3x3 median filter
    
    #Umbralizacion automatica de la imagen (saca mascara inicial)
    t, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    
    # obtener los contornos
    contours, _ = cv2.findContours(dst ,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    #Comprueba cual es el contorno mas grande
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    #Guardamos solo los contornos que sean mas grandes que un cierto porcentaje (marcado por threhold_area) del contorno mas grande 
    contours2=[contour for contour in contours if(cv2.contourArea(contour)>cv2.contourArea(biggest_contour)*threhold_area)]
    
    #Genera nueva mascara con los contornos obtenidos (ya no tendra outliers)
    mask = np.ones(image.shape[:2], dtype="uint8") * 255
    
    ## Draw the contours on the mask
    cv2.drawContours(mask, contours2, -1, 0, thickness =-1)
    
    #Recortamos la boundingbox del objeto para obtener una imagen y mascara con el objeto centrado
    x,y,w,h = cv2.boundingRect(mask)
    
        #Ampliamos un poco la boundingbox para evitar recortar mal imagen
    offset=max(int(x/10), int(y/10))
    x=x-offset; y=y-offset; w=w+(2*offset); h=h+(2*offset); 
    
    if(x<0): x=0 #Acotamos la ampliacion si se ha salido de la imagen
    if(y<0): y=0
    if((x+w)>(image.shape[1]-1)): w=(image.shape[1]-1)-x
    if((y+h)>(image.shape[0]-1)): h=(image.shape[0]-1)-y
    
    image_crop=image[y:y+h,x:x+w]
    mask_crop=mask[y:y+h,x:x+w]
    
    #Realizamos erosion de la mascara para que no obtenga borde blanco al aplicar la mascara con la imagen
    kernel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask_crop = cv2.erode(mask_crop.copy(),kernel,iterations = 5)
    
    #Sacamos la imagen recortada con la mascara aplicada
    if(muestra_resultado==True):
        print('Resultado recortado donde solo se encuentra el objeto:')
        image_segmented=cv2.bitwise_and(image_crop,image_crop, mask= mask_crop)
        segmentation=np.hstack([image_crop,image_segmented])
        plt.imshow(cv2.cvtColor(segmentation, cv2.COLOR_BGR2RGB))
        plt.show()
        print(mask_dir)
#        cv2.imshow("Image",image_segmented)
        
    #transformamos la mascara de 0s y 255s, a 0s y 1s
    mask_crop = mask_crop.astype(bool).astype('uint8') 
    
    return image_crop, mask_crop

_,_=genera_mascara(image, threhold_area=0.001, muestra_resultado=True)  
   