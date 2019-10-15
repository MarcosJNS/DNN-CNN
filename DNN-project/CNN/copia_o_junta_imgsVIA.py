# -*- coding: utf-8 -*-
"""
Script que copia de una carpeta a otra imagenes etiquetadas con la herramienta VIA 
(http://www.robots.ox.ac.uk/~vgg/software/via/), y en caso de que en la carpeta destino ya
hubiera este tipo de imagenes, se unen las etiquetas del JSON de la carpeta de origen al 
que ya existe en la carpeta destino.
"""
import os
import json
from shutil import copyfile

def save_obj_JSON(path_name , obj_to_save):
    """
    Funcion para guardar cualquier objeto python en un fichero .json
    
    path_name: ruta en la que se quiere guardar el archivo
    obj_to_save: objeto python que se quiere guardar con el nombre especificado
    """
    with open(path_name, 'w+', encoding="utf8") as f:  
        json.dump(obj_to_save, f, sort_keys=True, indent=4)
        
        
def load_obj_JSON(path_name):
    """
    Funcion para cargar un fichero .json en el que se había guardado un objeto de python
    
    path_name: ruta del fichero .json que se quiere cargar
    """
    with open(path_name, 'rb') as f:
        return json.load(f)
    

dataset_dir='asistente_dataset'
subset='val' #carpeta origen dodne estan las imagenes que se quieren copiar
destino='val2' #carpeta dentro de dataset_dir en la que se va a copiar las imagenes

dataset_dir_src = os.path.join(dataset_dir, subset)
dataset_dir_dst = os.path.join(dataset_dir, destino)

json_src=os.path.join(dataset_dir_src, "via_region_data.json")
json_dst=os.path.join(dataset_dir_dst, "via_region_data.json")

annotations_src = load_obj_JSON(json_src)
if(type(annotations_src) is dict):
    annotations_src = list(annotations_src.values())  # don't need the dict keys


# The VIA tool saves images in the JSON even if they don't have any
# annotations. Skip unannotated images.
annotations_src = [a for a in annotations_src if a['regions']]

#Copiamos en la carpeta de destino el JSON o lo unimos con uno ya existente
if(os.path.exists(json_dst)):
    annotations_dst=load_obj_JSON(json_dst)
    if(type(annotations_dst) is dict):
        annotations_dst = list(annotations_dst.values())  # don't need the dict keys
        
    annotations_dst = [a for a in annotations_dst if a['regions']] #Skip unannotated images
    annotations_dst.extend(annotations_src)
    save_obj_JSON(json_dst, annotations_dst)
else:
    copyfile(json_src, json_dst)


#Copiamos las imagenes etiquetadas:
for ann in annotations_src:
    
    img_src=os.path.join(dataset_dir_src, ann['filename'])
    img_dst=os.path.join(dataset_dir_dst, ann['filename'])
    copyfile(img_src, img_dst)
    