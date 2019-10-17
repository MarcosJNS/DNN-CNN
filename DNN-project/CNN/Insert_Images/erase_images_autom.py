# -*- coding: utf-8 -*-
"""
This script erases the images selected (from the dataset) and then updates the Json.
In case of mismatch between the Json and the datset use purgue_json.

"""
import os
from generate_images import load_obj_JSON, save_obj_JSON

path_dirTrain='train_steak'

imgs_to_delete=['',
                '',
                '',
                '']

#Cargamos las anotaciones en memoria:
    #Leemos los archivos JSON
path_annotations_json=os.path.join(path_dirTrain, 'annotations.json')
annotations_json = load_obj_JSON(path_annotations_json)


for image_i in imgs_to_delete:
    
    img_path=os.path.join(path_dirTrain, image_i)
    
    if(os.path.exists(img_path)):
        
        #Eliminamos la imagen
        os.remove(img_path)
        
        #Eliminamos las anotaciones de la imagen
        del annotations_json[image_i]
        
        #Guardamos las anotaciones sin las imagenes eliminadas
        save_obj_JSON(os.path.join(path_dirTrain, 'annotations.json'), annotations_json)
        
    else:
        print('La imagen indicada "{}" no existe'.format(image_i))