# -*- coding: utf-8 -*-
"""
Script para aplicarles blending a imagenes que ya habian sido creadas con la
aplicacion de pegado de objetos.
"""

import cv2
import numpy as np
import os
import json 
import shutil

#Importamos el codigo generar_imagenes.py para poder reconstruir imagenes creadas con ese programa
import generar_imagenes as gen_img


train_path='train' #train images to blend
new_train_path=train_path+'_blend'
if(not os.path.exists(new_train_path)):
    os.mkdir(new_train_path)

#Eliminamos de la lista el fichero de anotaciones    
list_imgs=sorted(os.listdir(train_path))
list_imgs.remove('annotations.json')

#copiamos las anotaciones en la nueva carpeta
shutil.copyfile(train_path+'/annotations.json', new_train_path+'/annotations.json')

# Load annotations
annotations = json.load(open(os.path.join(train_path, "annotations.json")))

for img_name in list_imgs:

    #leemos la anotacion de esa imagen:
    ann=annotations[img_name]
    
    #Extraemos los parametros necesarios para regenerar las imagenes:
    image_BG_path = ann['image_BG_path']
    images_FG_path=ann['images_FG_path']
    labels=ann['labels']
    pos_xy=ann['pos_xy']
    scales=ann['scale']
    height, width = ann['image_size']
    
    #Leemos la imagen de fondo
    image_BG=cv2.imread(image_BG_path)
    
    #Creamos un objeto de la clase imagen_resultado para poder almacenar el resultado y obtener las mascaras con las oclusiones ya 
    #tratadas
    img_res=gen_img.imagen_resultado(image_BG_path)
    
    #Reconstruimos la imagen creada para obtener tambien las mascaras de los objetos pegados:
    idx=np.arange(len(labels))   
    for i, path_image_i, label, p_xy, scale in zip(idx, images_FG_path, labels, pos_xy, scales):
        
        image_FG=cv2.imread(path_image_i)
    
        image_obj, mask_obj=gen_img.genera_mascara(image_FG)
        
        image_obj, mask_obj=gen_img.cambio_escala_obj(image_obj, mask_obj, scale=scale)
        
        image_masked, mask= gen_img.aplica_mascara(image_BG, image_obj, mask_obj, p_xy, blend_obj=True)
        
        image_BG=image_masked
    
    #Guardamos la nueva imagen con el blending aplicado:
    new_image_path=os.path.join(new_train_path, img_name)
    cv2.imwrite(new_image_path, image_masked)
    
    