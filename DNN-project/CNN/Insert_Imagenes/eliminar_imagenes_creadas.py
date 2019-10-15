# -*- coding: utf-8 -*-
"""
Script para eliminar las imagenes indicadas por su nombre en la lista imgs_to_delete

Elimina las imagenes de la carpeta y tambien sus etiquetas del script .JSON

"""
import os
from generar_imagenes import load_obj_JSON, save_obj_JSON

path_dirTrain='val'

imgs_to_delete=['20190404_20h45m23s_95.png',
                '20190404_20h46m09s_96.png',
                '20190404_20h47m13s_97.png',
                '20190404_20h48m17s_100.png']

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