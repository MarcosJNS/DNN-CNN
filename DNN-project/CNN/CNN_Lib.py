"""
Mask R-CNN
Entrenado con dataset de objetos de cocina para el desarrollo de un asistente de cocina

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)

------------------------------------------------------------

"""

import os
import sys
import shutil
import json
import numpy as np
import skimage.draw
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

import random
import warnings
from datetime import datetime

# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import visualize

#Importamos el codigo generar_imagenes.py para poder reconstruir imagenes creadas con ese programa
import Insert_Images.generate_images as gen_img

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class CNN_Config(Config):
    """Configuration for training on the train dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "asistente"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # Background + (person, bottle, plate)

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 500   #100 #Con esto no varia el numero de imagenes que mete en cada batch del entrenamiento, solo sirve para que en tensorboard se actualice
                               #antes o despues los datos. Al final de cada epoch calcula el error de validacion por lo que tmapoco poner muy bajo para no gastar mucho 
                              #tiempo de entrenamiento en calcular el error de validacion.
    VALIDATION_STEPS = 50 #50
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
    #Learning Rate
#    LEARNING_RATE=0.001
    
    #Learning Rate
    LEARNING_RATE=1e-4 #0.001,pero todo el mundo esta usando 1e-4
    
    #Modificamos la funcion display de config para que tambien pueda devolver la informacion de configuracion en 
    #una lista:
    def display(self, return_info=False):
        """Display or return Configuration values."""
        
        #Solo mostramos por pantalla si return_info es False
        if(return_info==False):
            print("\nConfigurations:")
        else:
            list_info=[]
            
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
               
                #Solo mostramos por pantalla si return_info es False
                if(return_info==False):
                    print("{:30} {}".format(a, getattr(self, a)))
                    
                else:
                    list_info.append("{:30} {}".format(a, getattr(self, a)))
        print("\n")
        
        if(return_info==True):
            return list_info
        
    
class AsistenteInferenceConfig(CNN_Config):
    """Configuration for validation on the val dataset.
    Derives from the AsistenteConfig class and overrides some values.
    """
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


############################################################
#  Dataset
############################################################

class CNN_Dataset(utils.Dataset): #Coge como clase padre la utils.Dataset, por lo que contendra sus mismos atributos y funciones más los añadidos o modificados

    ##################################################################
    #   CARGA DE IMAGENES DESDE DATASET ASISTENTE ETIQUETADO A MANO
    ##################################################################
    
    def load_asistentedataset(self, dataset_dir, subset, clases_a_entrenar=[]):
        """Load a subset of the hand-labeled Asistente dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        clases_a_entrenar: Para seleccionar que clases se desean entrenar de todas las etiquetadas.
                           Si se deja vacio se entrenara con todas las clases etiquetadas.
        
        """
        # Add classes. We have only one class to add.
        if(not clases_a_entrenar): #Si no le indicas nada en clases_a_entrenar introduce todas las clases del dataset
            self.add_class("asistente", 1, "bottle")
            self.add_class("asistente", 2, "pan")
            self.add_class("asistente", 3, "pot")
            self.add_class("asistente", 4, "knife")
            self.add_class("asistente", 5, "potato")
            self.add_class("asistente", 6, "glass")
            self.add_class("asistente", 7, "person")
            self.add_class("asistente", 8, "plate")
            self.add_class("asistente", 9, "egg")
            self.add_class("asistente", 10, "carrot")
            self.add_class("asistente", 11, "green_pepper")
            self.add_class("asistente", 12, "orange")
            self.add_class("asistente", 13, "fork")
            self.add_class("asistente", 14, "oven")
            self.add_class("asistente", 15, "microwave")
            self.add_class("asistente", 16, "spoon")
            self.add_class("asistente", 17, "washing_machine")
        
        else: #En caso contrario se le introducen las clases indicadas EN EL ORDEN DE APARICION DE LA LISTA:
            for i, class_name in enumerate(clases_a_entrenar):
                self.add_class("asistente", i+1, class_name) #i+1 por que la clase 0 es siempre 'BG'
                
                
        # Train or validation dataset?
#        assert subset in ["train", "val", "val2"], "Tipos de dataset validos: train, val, val2"
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        if(type(annotations) is dict):
            annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]
        
        #Comprobamos si se esta cargando el dataset de manos, y en ese caso se introduce un numero de imagenes
        #determinado (ya que hay muchas):
        if(subset=="train" and ("hand" in clases_a_entrenar)):
            num_max=200
            annotations=list(np.random.choice(annotations, num_max))
            
        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # Get the class name of each polygon (mask) of an object. This are stored
            # in the region_attributes (see json format above) in attribute defined as Class_name
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:            
                #En caso de que no se haya indicado ninguna clase a entrenar, se intoducen todas las mascaras etiquetadas
                if (not clases_a_entrenar):
                    polygons = [r['shape_attributes'] for r in a['regions'].values()]
                    class_names=[r['region_attributes']['class_name'] for r in a['regions'].values()]
                
                else:
                    #En caso de solo tener en cuenta las clases indicadas en clases_a_entrenar, aunque haya mas clases etiquetadas
                    #en cada imagen:
                    polygons=[]
                    class_names=[]
                    for r in a['regions'].values():
                        if(r['region_attributes']['class_name'] in clases_a_entrenar):
                            polygons.append(r['shape_attributes'])
                            class_names.append(r['region_attributes']['class_name'])
            else:     
                #En caso de que no se haya indicado ninguna clase a entrenar, se intoducen todas las mascaras etiquetadas
                if (not clases_a_entrenar): 
                    polygons = [r['shape_attributes'] for r in a['regions']]
                    class_names = [r['region_attributes']['class_name'] for r in a['regions']] 
                else:
                    #En caso de solo tener en cuenta las clases indicadas en clases_a_entrenar, aunque haya mas clases etiquetadas
                    #en cada imagen:
                    polygons=[]
                    class_names=[]
                    for r in a['regions']:
                        if(r['region_attributes']['class_name'] in clases_a_entrenar): #Solo guardara las mascaras de las clases indicadas
                            polygons.append(r['shape_attributes'])
                            class_names.append(r['region_attributes']['class_name'])
            
            #Si no hay ninguna mascara guardada significa que las clases indicadas no estan etiquetadas por lo 
            #que se lanza un error
            assert class_names, "Ninguna mascara guardada, imagen sin etiquetas"
            
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            #Add image in dataset class with the function of the parent class
            self.add_image(
                "asistente",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                class_names=class_names)

    def load_mask_asistente(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a asistente dataset image, delegate to parent class (y asi devolvera una mascara vacia).
        image_info = self.image_info[image_id]
        if image_info["source"] != "asistente":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count], and assign class ids to each mask
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        class_ids=np.zeros([mask.shape[-1]], dtype=np.int32)
        
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            
            #Como a veces indica tambien el pixel final del 'eje y' o 'eje x'  y no puede ser (porque empieza en cero),  se le indica 
            #que si el 'pto x/y' es igual al tamanno total de su eje que se le reste 1
            rr[rr >= mask.shape[0]] = mask.shape[0]-1 #eje y
            cc[cc >= mask.shape[1]] = mask.shape[1]-1 #eje x

            #Creamos la mascara con los pixeles indicados
            mask[rr, cc, i] = 1
            
            #Assign the classs id of the mask (the label):
            class_ids[i]=self.class_names.index(info["class_names"][i])

        # Return mask, and array of class IDs of each instance. 
        return mask.astype(np.bool), class_ids
        
    
    ##################################################################
    #     CARGA DE IMAGENES DESDE DATASET ASISTENTE SINTETICO
    ##################################################################
    
    def load_datasetSinteticoAsist(self, dataset_dir, subset, clases_a_entrenar=[]):
        """Load a subset of the sintetic Asistente dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        clases_a_entrenar: Para seleccionar que clases se desean entrenar de todas las etiquetadas.
                           Si se deja vacio se entrenara con todas las clases etiquetadas.
        
        """
        # Add classes. 
        if(not clases_a_entrenar): #Si no le indicas nada en clases_a_entrenar introduce todas las clases del dataset
            self.add_class("asistenteSint", 1, "bottle")
            self.add_class("asistenteSint", 2, "pan")
            self.add_class("asistenteSint", 3, "pot")
            self.add_class("asistenteSint", 4, "knife")
            self.add_class("asistenteSint", 5, "potato")
            self.add_class("asistenteSint", 6, "person")
            self.add_class("asistenteSint", 7, "egg")
            self.add_class("asistenteSint", 8, "orange")
        
        else: #En caso contrario se le introducen las clases indicadas EN EL ORDEN DE APARICION DE LA LISTA:
            for i, class_name in enumerate(clases_a_entrenar):
                self.add_class("asistenteSint", i+1, class_name) #i+1 por que la clase 0 es siempre 'BG'
                
                
        # Train or validation dataset?
#        assert subset in ["train", "train_auto", "train_blend", "val"], "Tipos de dataset validos: train, train_auto, train_blend, val"
        dataset_subset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        annotations = json.load(open(os.path.join(dataset_subset_dir, "annotations.json")))
            
        # Add images
        for img_name, ann in annotations.items():
                 
            #En caso de que no se haya indicado ninguna clase a entrenar, se intoducen todas las mascaras etiquetadas
            if (not clases_a_entrenar): 
                images_FG_path=ann['images_FG_path']
                labels=ann['labels']
                pos_xy=ann['pos_xy']
                scales=ann['scale']
                   
            else:
                #En caso de solo tener en cuenta las clases indicadas en clases_a_entrenar, aunque haya mas clases etiquetadas
                #en cada imagen:
                images_FG_path=[]
                labels=[]
                pos_xy=[]
                scales=[]
            
                for path_image_i, label, p_xy, scale in zip(ann['images_FG_path'], ann['labels'], ann['pos_xy'], ann['scale']):
                    if(label in clases_a_entrenar): #Solo guardara las mascaras de las clases indicadas
                        images_FG_path.append(path_image_i)
                        labels.append(label)
                        pos_xy.append(p_xy)
                        scales.append(scale)
            
            #Si no hay ninguna mascara guardada significa que las clases indicadas no estan etiquetadas por lo 
            #que esas imagenes no se introducen en el dataset (se sale del bucle)
            if(len(labels) == 0):
                print("{}: Ninguna mascara guardada, imagen sin etiquetas".format(img_name))
                continue
            
            
            height, width = ann['image_size']
            
            #Completamos las rutas con la direccion del dataset:
            image_path = os.path.join(dataset_subset_dir, img_name)
            image_BG_path = os.path.join(dataset_dir, ann['image_BG_path'])
            images_FG_path=[os.path.join(dataset_dir, img_FG_path) for img_FG_path in images_FG_path]
            
            #Add image in dataset class with the function of the parent class
            self.add_image(
                "asistenteSint",
                image_id=img_name,  # use img name as a unique image id
                path=image_path,
                width=width, height=height,
                image_BG_path=image_BG_path,
                images_FG_path=images_FG_path,
                labels=labels,
                pos_xy=pos_xy,
                scales=scales)
            
    def load_mask_asistenteSint(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a asistente dataset image, delegate to parent class (y asi devolvera una mascara vacia).
        image_info = self.image_info[image_id]
        if image_info["source"] != "asistenteSint":
            return super(self.__class__, self).load_mask(image_id)

        #Inicializamos el vector de 
        class_ids=np.zeros([len(image_info["labels"])], dtype=np.int32)
        
        #Leemos la imagen de fondo necesaria para reconstruir la imagen:
        image_BG=cv2.imread(image_info['image_BG_path'])
        
        #Creamos un objeto de la clase imagen_resultado para poder almacenar el resultado y obtener las mascaras con las oclusiones ya 
        #tratadas
        img_res=gen_img.imagen_resultado(image_info['image_BG_path'])
        
        #Reconstruimos la imagen creada para obtener tambien las mascaras de los objetos pegados:
        idx=np.arange(len(image_info['labels']))   
        for i, path_image_i, label, p_xy, scale in zip(idx, image_info['images_FG_path'], image_info['labels'], image_info['pos_xy'], image_info['scales']):
            
            image_FG=cv2.imread(path_image_i)

            image_obj, mask_obj=gen_img.genera_mascara(image_FG)
            
            image_obj, mask_obj=gen_img.cambio_escala_obj(image_obj, mask_obj, scale=scale)
            
            image_masked, mask= gen_img.aplica_mascara(image_BG, image_obj, mask_obj, p_xy)
            
            #Se ha obtenido la mascara del objeto pero sin tener en cuenta las oclusiones, por lo que hay que utilizar la clase
            #imagen_resultado para poder eliminar de las mascaras los pixeles ocluidos por otros objetos. Una vez se hayan guardado 
            #todas mascaras se podra extraer de img_res las mascaras finales
            img_res.nuevo_objeto(path_image_i, image_masked, mask, label, p_xy, scale)
            
            image_BG=image_masked
            
            #Assign the classs id of the mask (the label):
            class_ids[i]=self.class_names.index(label)
        
        #transformamos las mascaras obtenidas a array numpy en el formato [height, width, instance_count]
        masks=np.asarray(img_res.masks_FG)
        masks=np.moveaxis(masks, 0, -1) #Para dejarlo en formato [height, width, instance_count] en vez de [instance_count, height, width]
        
        # Return mask, and array of class IDs of each instance. 
        return masks.astype(np.bool), class_ids
    
    
    ################################################
    #   CARGA DE IMAGENES DESDE DATASET COCO
    ################################################
    
    
    def load_coco(self, dataset_dir, subset, year='2017', class_ids=None, num_imagesPorClase=None,
              class_map=None, return_coco=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        """

        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                
                if(num_imagesPorClase != None):
                    random.seed(42) #Para que siempre escpja las mismas images (aunque sea aleatoriamente)
                    image_ids.extend(random.sample(list(coco.getImgIds(catIds=[id])), num_imagesPorClase))
                else:
                    image_ids.extend(list(coco.getImgIds(catIds=[id])))
                    
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=i, catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco


    def load_mask_coco(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(self.__class__, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(self.__class__, self).load_mask(image_id)


    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

    ########################################################################################
    #   DEFINICION DE FUNCION load_mask() (OBLIGATORIA) EN FUNCION DE QUE DATASET PROVENGA
    ########################################################################################
    
    def load_mask(self, image_id):
        """Generate instance masks for an image.
        
        IMPORTANTE:
        La funcion load_mask() es llamada cuando se va a cargar una imagen del dataset justo antes
        de pasarsela a la red, en la función load_image_gt() declarada en línea 1187 del fichero model.py.
        Por eso no se puede cambiar de nombre, sino que aunque cojas imagenes de dos Dataset distintos
        te creas una funcion load_mask_dataset para cada uno, y luego desde esta funcion indicas cual de 
        esas funciones llamar para devolver las mascara y las etiquetas dependiendo del ["source"] (del dataset)
        del que provengan
        
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a asistente dataset image, delegate to parent class (y asi devolvera una mascara vacia).
        image_info = self.image_info[image_id]
        if image_info["source"] == "asistente":
            return self.load_mask_asistente(image_id)
        
        elif image_info["source"] == "asistenteSint":
            return self.load_mask_asistenteSint(image_id) 
        
        elif image_info["source"] == "coco":
            return self.load_mask_coco(image_id) 
        
        else:
            return super(self.__class__, self).load_mask(image_id)

    ##################################################################################################################
    #   FUNCION PARA HACER UN CONTEO DE IMAGENES CON ETIQUETAS DE CADA CLASE, Y CONTEO DE ETIQUETAS DE CADA CLASE
    ##################################################################################################################
    
    def datasetCompletoInfo(self, return_info=False):
        """
        Teniendo ya cargado TODO el dataset y DESPUES de haber llamado a la funcion "prepare()",
        al ejecutar esta funcion, hará un print de:
            - Por cada clase, el numero de imagenes en las que sale un elemento de esa clase
            - Por cada clase, el numero total de elementos de esa clase que han salido en la totalidad de imagenes
            
        Si return_info=True, devuelve la informacion del dataset en una lista
        """
        
         #Creamos el diccionario que hara de contador de imagenes en las que aparece cada clase una o mas veces .
        contador_imagenes=dict()
        
        #Creamos el diccionario que hara de contador de elementos de cada clase en la totalidad de imagenes. 
        contador_elementos=dict()
        
        #Inicializamos los diccionarios introduciendoles como clave de cada contador las id de las clases
        for class_id in self.class_ids:
            contador_imagenes[str(class_id)] = 0 #Inicializamos a cero cada contador
            contador_elementos[str(class_id)] = 0 #Inicializamos a cero cada contador
            
        #Para cada imagen del dataset, realizamos el conteo de elementos e imagenes de cada clase
        for image_info_i in self.image_info:
            
            class_ids_en_la_imagen=[] 
            
            #En caso de que la imagen venga del dataset coco:
            if(image_info_i['source'] == 'coco'): 
                
                for ann_i in image_info_i['annotations']:
#                    class_id= ann_i['category_id'] #Solo para debug
                    #En caso de que en la imagen no se haya observado todavia esta clase, se introduce en "class_ids_en_la_imagen"
                    #y se suma uno al contador de imagenes de esa clase:
                    if(ann_i['category_id'] not in class_ids_en_la_imagen):
                        
                        class_ids_en_la_imagen.append(ann_i['category_id'])
                        contador_imagenes[str(ann_i['category_id'])]= contador_imagenes[str(ann_i['category_id'])] +1
                    
                    #Se le suma uno al contador de elementos de la clase correspondiente:
                    contador_elementos[str(ann_i['category_id'])]= contador_elementos[str(ann_i['category_id'])] +1
              
            #En caso de que la imagen venga del dataset asistente o asistenteSint 
            #(solo cambia entre los dos el nombre de la variable donde se guardan las etiquetas):
            elif(image_info_i['source'] == 'asistente' or image_info_i['source'] == 'asistenteSint'):
                    
                if(image_info_i['source']== 'asistente'):
                    labels=image_info_i['class_names']
                    
                elif(image_info_i['source'] == 'asistenteSint'):
                    labels=image_info_i['labels']
                
                for etiqueta in labels:
                    
                    #Transformamos la etiqueta al id correspondiente:
                    class_id=self.class_names.index(etiqueta)
                    
                    #En caso de que en la imagen no se haya observado todavia esta clase, se introduce en "class_ids_en_la_imagen"
                    #y se suma uno al contador de imagenes de esa clase:
                    if(class_id not in class_ids_en_la_imagen):
                        
                        class_ids_en_la_imagen.append(class_id)
                        contador_imagenes[str(class_id)]= contador_imagenes[str(class_id)] +1
                    
                    #Se le suma uno al contador de elementos de la clase correspondiente:
                    contador_elementos[str(class_id)]= contador_elementos[str(class_id)] +1
                    
        #Lista en la que se almacenara la informacion recopilada en caso de que return_info=True:
        list_info=[]      
        
        #Visualizamos los contadores de imagenes y de clases:
        print('\nCONTADORES DE IMAGENES POR CLASE:\n')
        if(return_info==True):
            list_info.append('\nCONTADORES DE IMAGENES POR CLASE:\n')
            
        for class_id, contador in contador_imagenes.items():
            if(class_id != '0'): #La clase BG siempre es 0 porque no la hemos contado (siempre seria igual al num de imagenes)
                print('    clase: {}  num_imagenes:  {}'.format(self.class_names[int(class_id)], contador))
                
                if(return_info==True):
                    list_info.append('    clase: {}  num_imagenes:  {}'.format(self.class_names[int(class_id)], contador))
            
            
        print('\nCONTADORES DE ELEMENTOS TOTALES POR CLASE:\n')
        if(return_info==True):
            list_info.append('\nCONTADORES DE ELEMENTOS TOTALES POR CLASE:\n')
            
        for class_id, contador in contador_elementos.items():
            if(class_id != '0'): #La clase BG siempre es 0 porque no la hemos contado (siempre seria igual al num de imagenes)
                print('    clase: {}  num_elementos:  {}'.format(self.class_names[int(class_id)], contador))
                
                if(return_info==True):
                    list_info.append('    clase: {}  num_elementos:  {}'.format(self.class_names[int(class_id)], contador))
   
        if(return_info==True):
            return list_info
        
        
    #################################################################################################################################
    #   REDEFINICION DE FUNCION 'prepare()' DE 'utils.Dataset' PARA QUE SE PUEDAN FUSIONAR DOS CLASES IGUALES DE DISTINTOS DATASET 
    #################################################################################################################################                               
    
    def prepare(self, class_map=None):
        
        """Prepares the Dataset class for use.

        class map: a list with the names of the classes to merge. When class_map is not None, it handle mapping
                   classes from different datasets to the same class ID.
        """
        
        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        #Fusion de las clases iguales provenientes de datasets distintos que se han indicado en class_map mediante la fusion de los
        #id del dataset global
        if(class_map != None):
            
            for class_i in class_map: #Para cada una de las clases indicadas en class_map
                
                #comprueba los indices donde se encuentra en 'class_names' el nombre de clase 'class_i' (hay que convertirlo a array numpy para poder usar np.where)
                idx_class=list(np.where(class_i==np.array(self.class_names))[0]) 
                
                #En caso de que la clase indicada no se encuentre en dos dataset distintos se indica un warning y no se hace nada con los ids 
                if(len(idx_class)<2):
                    warnings.warn("La clase '{}' no aparece en dos sources distintos".format(class_i))
                    break
                
                #Transformamos los distintos ID de la clase a fusionar a uno solo, que sera igual al primero de los id que aparecen:
                id_class_merged=self.class_ids[idx_class[0]]
                
                for i in range(1, len(idx_class)): 
                    self.class_ids[idx_class[i]]= id_class_merged
                    self.num_classes=self.num_classes-1
                                        

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id        #CUANDO CARGA LAS MASCARAS LEE class_from_source_map PARA ASIGNAR LA id DEL DATASET GLOBAL(NO LA DEL source)
                                      for info, id in zip(self.class_info, self.class_ids)} #POR ESO SI SE INDICA ALGUNA CLASE A FUSIONAR EN 'class_map' HABRA QUE MODIFICAR ANTES DE ESTO LAS class_ids
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}
        
        #En caso de haber fusionado alguna clase, se eliminan los id y nombres repetidos (para poder crear class_from_source_map no se ha hecho antes)
        if(class_map != None):
            
            #Eliminamos ids repetidos:
            class_ids=list( self.class_ids) #Transformamos a lista el array numpy
            class_ids=sorted(set(class_ids), key=class_ids.index) #Con esto se queda solo con el primero de los elementos repetidos en el mismo orden que estaba
            self.class_ids=np.array(class_ids) #Volvemos a pasar a array numpy
            
            #Eliminamos class_names repetidos:
            self.class_names=sorted(set(self.class_names), key=self.class_names.index)   
        
        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0:
                    self.source_class_ids[source].append(i)
                    
                elif source == info['source']:
                    id_global=self.map_source_class_id('{}.{}'.format(info['source'], info['id']))
                    self.source_class_ids[source].append(id_global)
            
            self.source_class_ids[source]=sorted(self.source_class_ids[source])
            
            
    #################################################################################################################################
    #   REDEFINICION DE FUNCION 'image_reference()' DE 'utils.Dataset' PARA QUE MUESTRE LA RUTA DE LA IMAGEN DADO SU id
    #################################################################################################################################         
        
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "asistente" or info["source"] == "coco":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)    
            
            
#%% 
##########################################
#FUNCIONES CREADAS PARA VISUALIZACION
##########################################
            
def draw_box(img, pt1_box, pt2_box, class_name, score, color, fontScale=2, alpha=0.7, grosor_box=10, grosor_texto=10, offset_texto=50):
    
    # create two copies of the original image -- one for
    # the overlay and one for the final output image
    overlay = img.copy()
    output = img.copy()
    
    # draw a red rectangle 
    cv2.rectangle(overlay, pt1=pt1_box, pt2=pt2_box, color=color, thickness=grosor_box)
        
    #draw text
    cv2.putText(overlay, text="{}: {:.4f}".format(class_name, score), org=(pt1_box[0],pt1_box[1]-offset_texto), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale, color=color, thickness=grosor_texto)
        		
    # apply the overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        
    return output

def pinta_contorno(img, posiciones, color, radio_vecinos=4):
    """
    funcion que pinta en una imagen dada el controno de una mascara dadas las posiciones x e y del mismo
    """
    for c,color_i in enumerate(color):
        x=list(posiciones[:,0])
        
        y=list(posiciones[:,1])
        
        for i in range(len(x)):
            img[y[i]-radio_vecinos:y[i]+radio_vecinos, x[i]-radio_vecinos:x[i]+radio_vecinos, c]= int(color_i * 255)
            
    return img




def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16),
                      show_mask=True, show_bbox=True,
                      colors=None):
    """
    Funcion con la que se visualiza el resultado obtenido por mask RCNN utilizando matplotlib y transformandola
    finalmente a un array numpy que puede ser visualizado por openCV
    
    ----------------------
    Parametros de entrada:
    ----------------------
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    display_det: (optional) Lista de booleanos indicando que detecciones se quieren visualizar y cuales no
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    height, width = image.shape[:2]

    fig, ax = plt.subplots(1,frameon=False)

    ax.set_axis_off()
        
    # Generate random colors
    colors = colors or visualize.random_colors(N)
    # Show area outside image boundaries.

    ax.axis('off')
        
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        class_id = class_ids[i]

        color = colors[class_id-1]

        # Bounding box
        if(not np.any(boxes[i])):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = visualize.patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        score = scores[i] if scores is not None else None
        label = class_names[class_id]

        caption = "{} {:.3f}".format(label, score) if score else label
            
        ax.text(x1, y1 + 8, caption,
                color='w', size=16, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = visualize.apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = visualize.find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = visualize.Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))

    # To transform the drawn figure into ndarray X
    fig.tight_layout()
    fig.canvas.draw()
    X = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    X = X.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    #Para eliminar los bordes blancos:
    idx_limL=0
    idx_limR=-1
    idx_limU=0
    idx_limD=-1
    
    while(False not in (X[int(X.shape[0]/2), idx_limL]==[255,255,255])):
        idx_limL=idx_limL+1
        
    while(False not in (X[int(X.shape[0]/2), idx_limR]==[255,255,255])):
        idx_limR=idx_limR-1
    
    while(False not in (X[idx_limU, int(X.shape[1]/2)]==[255,255,255])):
        idx_limU=idx_limU+1
        
    while(False not in (X[idx_limD, int(X.shape[1]/2)]==[255,255,255])):
        idx_limD=idx_limD-1
        
    X=X[idx_limU:idx_limD, idx_limL:idx_limR]
        
    #Para que la imagen que devuelve sea del mismo tamanno que la de entrada:
    X=cv2.resize(X, (width,height), interpolation=cv2.INTER_LANCZOS4)

    # open cv's RGB style: BGR
    plt.close()
    return X

        
        
        
        
def display_instances2(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):
    """
    Funcion para visualizar utilizando solo openCV
    
    ----------------------
    Parametros de entrada:
    ----------------------
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # Generate random colors
    colors = colors or visualize.random_colors(N) #Si colors no tiene valor, utilizara la funcion visualize.random_colors(N)
                                                  #Si colors si que tiene valor, utilizara ese ya que es la primera opcion que se le da

    #Miramos si la imagen leida esta en formato 0 a 255:
    assert np.dtype(image[0,0,0]).type is np.uint8, "Imagen no tiene formato uint8 [de 0 a 255]"

    masked_image = image.astype(np.uint8).copy()
    
    
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
            
        #cogemos el num de pixeles del lado mas grande de la imagen (el mayor de alto o ancho)
        #y obtenemos autmaticamente el grosor de la BB
        size_img=np.max(np.shape(masked_image))
        
        grosor_box=int(np.round(size_img/400))*2
        
        if show_bbox:
            
            #Para saber que grosor y offset ponerle a la letra automaticamente dependiendo de tamanno de imagen:
            if(size_img<1000):
                fontScale=0.5
                grosor_texto=2
                offset_texto=10
                
            elif(size_img<2000):
                fontScale=1.5
                grosor_texto=4
                offset_texto=25  
                
            elif(size_img<5000):
                fontScale=4
                grosor_texto=10
                offset_texto=50  
                
            else:
                fontScale= grosor_box/5
                grosor_texto=int(grosor_box/1.5) if int(grosor_box/1.5)>=1 else 1
                offset_texto=grosor_box*2
                print("otro")
            
                
            masked_image=draw_box(masked_image, (x1, y1), (x2, y2), label, score, color=np.array(color)*255, alpha=0.7, fontScale=fontScale, grosor_box=grosor_box, grosor_texto=grosor_texto, offset_texto=offset_texto)

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = visualize.apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        
        contours = visualize.find_contours(padded_mask, 0.5)
        
        #Para saber que grosor ponerle al contorno automaticamente dependiendo de tamanno de imagen:
        #radio_vecinos=int(np.round(size_img/400))
        radio_vecinos=int(np.round(grosor_box/4)) if int(np.round(grosor_box/4))>0 else 1
        
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1            
            verts=verts.astype('int')
            
            masked_image=pinta_contorno(masked_image, verts, color, radio_vecinos=radio_vecinos)
        
    return masked_image.astype(np.uint8)

def pltFig2RGBNumpy ( fig ):
    """
    @brief Convert a Matplotlib figure to a 3D numpy array with RGB channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
    X = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    X = X.reshape(fig.canvas.get_width_height()[::-1] + (3,))
 
    plt.close(fig)
    return X

#%% 
##############################################################
#FUNCION CREADA PARA GUARDAR LA INFORMACION DEL ENTRENAMIENTO
##############################################################
    
def save_train_information(path_txt_file, type_of_info, object_info, file_new):
    """
    Funcion con la que se puede guardar tres tipos de informacion de un entrenamiento realizado en un fichero .txt
    
    ----------------------
    Parametros de entrada:
    ----------------------
    path_txt_file: ruta del fichero de texto que queremos leer/escribir
    type_of_info: pueder ser:'config_info' --> para guardar la informacion de configuracion de la red
                             'dataset_info' --->  para guardar la informacion del dataset utilizado
                             'stages_train_info' ----> para guardar la informacion del entrenamiento realizado (capas entrenadas, epoch, LR)
                             
    object_info: objeto python necesario para la informacion seleecionada con type_of_info: 
                 - si type_of_info= 'config_info' --> object_info tiene que ser el objeto creado config
                 - si type_of_info= 'dataset_info' --> object_info tiene que ser una lista con el datataset de entrenamiento y el de validacion([dataset_train, dataset_val])
                 - si type_of_info= 'stages_train_info' --> object_info tiene que ser un lista con la siguiente informacion: [layers, epochs, LR]
                 - si type_of_info= 'augmentations' --> object_info tiene que ser un objeto de la clase imgaug
                 
    file_new: booleano para indicar si se quiere crear un nuevo fichero o leer uno anteriormente guardado con el nombre indicado
    
    ----------------------
    Parametros de salida:
    ----------------------
    ruta del archivo txt guardado. Si no ha cambiado devuelve la misma de entrada
    """
    
    assert type_of_info=='config_info' or type_of_info=='dataset_info' or type_of_info=='stages_train_info' or type_of_info=='augmentations', 'el tipo de informacion type_of_info indicado no es valido, tiene que ser: config_info, dataset_info, stages_train_info, o augmentations'
    
    #Si el fichero no existe, se crea poniendole en primer lugar el timestamp:
    if(not os.path.exists(path_txt_file)):
        
        now = datetime.now()
        timestamp = now.strftime("\n%d/%m/%Y - %Hh%Mm%Ss\n")
        
        with open(path_txt_file, 'a+') as file:  
            file.write(timestamp)
    
    #Si si que existe pero no queremos sobreescribir un fichero previo, se cambia el nombre a poner al fichero
    #para crear uno nuevo y se le introduce el timestamp
    elif(os.path.exists(path_txt_file) and file_new==True):
        
        #Cambiamos el nombre a poner al nuevo fichero
        num=0
        path_txt_file2=path_txt_file[:]
        while(os.path.exists(path_txt_file2)):
           num=num+1 
           path_txt_file2= '../..'+path_txt_file.split('.')[-2]+('_{}.txt'.format(num))
       
        path_txt_file=path_txt_file2[:]
        
        #Creamos el fichero y le ponemos el timestamp
        now = datetime.now()
        timestamp = now.strftime("\n%d/%m/%Y - %Hh%Mm%Ss\n")
        with open(path_txt_file, 'a+') as file:  
            file.write(timestamp)
        
        
    if(type_of_info == 'config_info'):
        
        lines_to_write= ['\n\n',
                         '------------------------------------------------------------------------------------------------\n',
                         '                         INFORMACION DE CONFIGURACION DE LA RED                                 \n',
                         '------------------------------------------------------------------------------------------------\n'] 
        
        with open(path_txt_file, 'a+') as file:  
            
            file.writelines(lines_to_write) 
            
            info_config=object_info.display(return_info=True)
            
            for line in info_config:
                file.write(line+'\n') 
        
    elif(type_of_info == 'dataset_info'):
        
        lines_to_write= ['\n\n',
                         '------------------------------------------------------------------------------------------------\n',
                         '                    INFORMACION DE DATASET DE ENTRENAMIENTO/VALIDACION                          \n',
                         '------------------------------------------------------------------------------------------------\n']
        
        dataset_train=object_info[0]
        dataset_val=object_info[1]
        
        with open(path_txt_file, 'a+') as file:  
        
            file.writelines(lines_to_write) 
            
            #Escribimos el nombre de las clases de los dataset con las que se ha entrenado:
            file.write('\nclass_names={}\n'.format(str(dataset_train.class_names)))
            
            #Escribimos el conteo de imagenes y de objetos en el dataset train:
            info_dataset_train=dataset_train.datasetCompletoInfo(return_info=True)
            
            file.write('\nInformacion dataset ENTRENAMIENTO--------------\n')
            
            file.write("\nNUMERO IMAGENES DATASET TRAIN: {}\n".format(len(dataset_train.image_ids)))
            
            for line in info_dataset_train:
                file.write(line+'\n') 
            
            #Escribimos el conteo de imagenes y de objetos en el dataset val:
            info_dataset_val=dataset_val.datasetCompletoInfo(return_info=True)
            
            file.write('\nInformacion dataset VALIDACION--------------\n')
            
            file.write("\nNUMERO IMAGENES DATASET VAL: {}\n".format(len(dataset_val.image_ids)))
            for line in info_dataset_val:
                file.write(line+'\n')
                
    elif(type_of_info == 'stages_train_info'):
        
        lines_to_write= ['\n\n',
                         '------------------------------------------------------------------------------------------------\n',
                         '                             STAGE DE ENTRENAMIENTO REALIZADA                                   \n',
                         '------------------------------------------------------------------------------------------------\n']
        
        list_info=['layers= {}\n'.format(object_info[0]),
                   'epochs= {}\n'.format(object_info[1]),
                   'learning_rate= {}\n'.format(object_info[2])]
        
        with open(path_txt_file, 'a+') as file:  
            
            file.writelines(lines_to_write) 
            
            file.writelines(list_info) 
            
    elif(type_of_info == 'augmentations'):
        
        name_pkl='\\'.join(path_txt_file.split(os.sep)[:-1])+'\\augmentations.pkl'
        
        print('\n\n',
                         '------------------------------------------------------------------------------------------------\n',
                         '                                        AUGMENTATIONS                                           \n',
                         '------------------------------------------------------------------------------------------------\n',
                         'augmentations saved as pkl in {}\n'.format(name_pkl))
                         

        with open(name_pkl, 'wb') as handle:
            pickle.dump(object_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
    
    return path_txt_file
     
#%% 
##############################################################
#FUNCIONES PARA CALCULAR LA METRICA mAP SOBRE UN DATASET
##############################################################        
 
def mAP_conjunto_deUnDataset(model_inference, dataset, num_images=None):
    """
    Funcion que calcula el AP de cada imagen por separado y te devuelve el mAP siendo este el promedio 
    del AP de todas las imagenes (no lo saca por clases)
    
    """
       
    # Compute VOC-Style mAP @ IoU=0.5 ---> mAP mean of all clases
    # Running on 'num_images' images. Increase for better accuracy.
    
    #If num_images is None, compute mAP over all dataset:
    if(num_images !=None):
        image_ids = np.random.choice(dataset.image_ids, num_images)
    else:
        image_ids=dataset.image_ids
    
    
    APs = []
    for image_id in image_ids:
        
        # Load image and ground truth data
        image = dataset.load_image(image_id)
        gt_mask, gt_class_id = dataset.load_mask(image_id)
        # Compute Bounding box
        gt_bbox = utils.extract_bboxes(gt_mask)
            
        # Run object detection
        results = model_inference.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                                             r["rois"], r["class_ids"], r["scores"], r['masks'])
            
        APs.append(AP)
    
    mAP=np.mean(APs)
    print("mAP: ", mAP)
    return mAP
        
    
def mAP_porClase_deUnDataset(model_inference, dataset, num_images=None):
       
    """
    Funcion mediante la cual se calcula el mAP de cada clase y el general del algoritmo.
    
    Para ello utiliza la libreria descargada mAP-master (https://github.com/Cartucho/mAP#running-the-code).
    Para poder calcular el mAP mediante esta libreria hay que dejar las etiquetas GT y las predicciones realizadas
    por el modelo en un formato especificado.
    El resultado obtenido tras ejecutar esta libreria queda almacenado en la carpeta mAP-master/results
    
    ----------------------
    Parametros de entrada:
    ----------------------
    model_inference: modelo de la red en modo inferencia con los pesos entrenados ya cargados
    dataset: objeto de la clase dataset con el dataset sobre el que se quiere calcular el mAP (normalmente el de validacion o test)
    num_images= parámetro opcional para seleccionar solo un numero determinado de fotos del dataset para calcular el mAP
    
    ----------------------
    Parametros de salida:
    ----------------------
    no tiene parametros de salida ya que los resultados obtenidos se almacenan en la carpeta mAP-master/results
    """
    
    #####################################################################################################################################
    #ELIMINAMOS EL CONTENIDO ANTERIOR DE LAS CARPETAS DONDE SE HAN DE ALMACENAR LOS .txt CON LAS ETIQUETAS Y LAS PREDICCIONES REALIZADAS
    #####################################################################################################################################
    output_GT_path='mAP-master/input/ground-truth'
    output_detections_path='mAP-master/input/detection-results'
    output_images_path='mAP-master/input/images-optional'

    if(len(os.listdir(output_GT_path))>0):
        shutil.rmtree(output_GT_path)
        shutil.rmtree(output_detections_path)
        shutil.rmtree(output_images_path)
        os.mkdir(output_GT_path)
        os.mkdir(output_detections_path)
        os.mkdir(output_images_path)
    
    ############################################################################################################
    #PREPARAMOS TANTO LAS ETIQUETAS GT COMO LAS PREDICCIONES EN EL FORMATO REQUERIDO (GUARDANDOLOS EN UN .txt)
    ############################################################################################################
    #If num_images is None, compute mAP over all dataset:
    if(num_images !=None):
        image_ids = np.random.choice(dataset.image_ids, num_images)
    else:
        image_ids=dataset.image_ids
    
    
    for image_id in image_ids:
        
        #load image and masks
        image = dataset.load_image(image_id)
        masks, gt_class_id = dataset.load_mask(image_id)
        # Compute Bounding box
        gt_bbox = utils.extract_bboxes(masks)
        
        #save image in images folder in mAP-master
        image_name=dataset.image_reference(image_id).split(os.sep)[-1]
        image_BGR=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_images_path,image_name), image_BGR)
        
        #Write GT info in GT folder in mAP-master
        output_GT_file=os.path.join(output_GT_path, image_name.split('.')[0])+'.txt'
        with open(output_GT_file, 'w+') as out_f:
            for class_id, bbox in zip(gt_class_id, gt_bbox): #Para cada una de las bounding box
                
                # Here we are dealing with ground-truth annotations
                # <class_name> <left> <top> <right> <bottom> [<difficult>]
                # todo: handle difficulty
                y_min, x_min, y_max, x_max  = bbox
                out_box = '{} {} {} {} {}'.format(dataset.class_names[class_id], x_min, y_min, x_max, y_max)
                out_f.write(out_box + "\n")
                
        # Run object detection
        with tf.device('/gpu:0'):
            results = model_inference.detect([image], verbose=0)
            r = results[0]
        
        #Write detection info in detection folder in mAP-master
        output_detection_file=os.path.join(output_detections_path, image_name.split('.')[0])+'.txt'
        with open(output_detection_file, 'w+') as out_f:
            for class_id, score, bbox in zip(r["class_ids"], r["scores"], r["rois"]): #Para cada una de las bounding box
                
                # Here we are dealing with ground-truth annotations
                # <class_name> <left> <top> <right> <bottom> [<difficult>]
                # todo: handle difficulty
                y_min, x_min, y_max, x_max = bbox
                out_box = '{} {:f} {} {} {} {}'.format(dataset.class_names[class_id], score, x_min, y_min, x_max, y_max)
                out_f.write(out_box + "\n")
         
            
    ####################################################################################
    #CALCULAMOS LA METRICA mAP DE CADA UNA DE LAS CLASES MEDIANTE LA LIBRERIA mAP-master
    ####################################################################################
    
    #Ejecutamos el script main.py como si lo pusieramos desde un terminal de comandos, indicandole ahi los argumentos deseados
    #(mirar el script main.py dentro de la carpeta mAP-master para ver todos los argumentos disponibles)
    ok=None
    while(ok!=0): #ejecutamos hasta que se realice correctamente
        ok=os.system("python mAP-master/main.py") #devuelve un 0 cuando se ha ejecutado correctamente
#        ok=os.system("python mAP-master/main.py --ignore potato") #Para que no tenga en cuenta la clase potato al calcular el mAP. Si se quieren ignorar mas clases se ponen a continuacion

    