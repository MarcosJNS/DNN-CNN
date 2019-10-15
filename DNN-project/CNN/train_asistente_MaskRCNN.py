# -*- coding: utf-8 -*-
"""
SCRIPT PARA ENTRENAMIENTO DE MASK-RCNN CON DATASET PARA EL ASISTENTE

"""

#%% 
############################################################
#  Preparacion para el entrenamiento
############################################################

#---------------------------
# Importacion de librerias:
# --------------------------
import os
import sys
import time
import numpy as np

from imgaug import augmenters as iaa


from asistente import AsistenteConfig as asistenteTrainConfig
from asistente import AsistenteDataset, save_train_information

ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library

from mrcnn import model as modellib

#-----------------------------
# Inrtroduccion de parametros:
# ----------------------------

# Root directory of the project


# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


logs_dir="../../logs"

#weights_pretrained='comparacion_blending/1con_blending/mask_rcnn_asistente_0100.h5'  #"coco", "imagenet", "last", "INTRODUCIR path"
weights_pretrained='last' #si pones last busca automaticamente los ultimos pesos entrenados en la carpeta logs
#weights_pretrained='mask_rcnn_asistente_sinNaranjas.h5'

#------------------------------------
# Creacion de configuracion y modelo:
# -----------------------------------

# Configurations
config = asistenteTrainConfig()
config.display()

# Create model
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=logs_dir)


#-------------------------------------------------------------
# Seleccion de pesos de los que partir para el entrenamiento:
# ------------------------------------------------------------

# Select weights file to load
if weights_pretrained == "coco":
    model_path = COCO_MODEL_PATH
elif weights_pretrained == "last":
    # Find last trained weights
    model_path = model.find_last()
elif weights_pretrained == "imagenet":
    # Start from ImageNet trained weights
    model_path = model.get_imagenet_weights()
    
else:
#    raise ValueError('"{}" no es uno de los modelos guardados'.format(weights_pretrained))
    model_path = logs_dir+'/'+weights_pretrained
    
#Si se va a entrenar el modelo con un numero de clases distinto al que tenia el modelo pre-entrenado del que
#se van a cargar sus pesos como punto de partida, se han de excluir las ultimas capas al cargar esos pesos
#ya que el numero de clases de salida no coincidira y nos daria un error:
different_pretrainedModel_numClass=False

if(weights_pretrained=='coco' or weights_pretrained=='imagenet'): #asumimos que si partes de pesos de COCO o IMAGENET vas a cambiar el numero de clases de salida
    different_pretrainedModel_numClass=True
    
if different_pretrainedModel_numClass == True:

    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
else:
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

#-----------------------------
# Carga del dataset deseado:
# ----------------------------
    
# Train or evaluate   
    
dataset_coco=os.path.join(ROOT_DIR,"samples/coco/coco_dataset")
#dataset_asistente=os.path.join(ROOT_DIR,"samples/asistente/asistente_dataset")
dataset_asistente=os.path.join(ROOT_DIR,"codigo_python/asistente/asistente_dataset")
#dataset_asistenteSint=os.path.join(ROOT_DIR,"samples/asistente/InsertarImagenes")
dataset_asistenteSint=os.path.join(ROOT_DIR,"codigo_python/asistente/InsertarImagenes")
#dataset_hands = os.path.join(ROOT_DIR, "samples/asistente/handDataset")
dataset_hands = os.path.join(ROOT_DIR, "codigo_python/asistente/handDataset")

# Training dataset. Use the training set and 35K from the
# validation set, as as in the Mask RCNN paper.
dataset_train=AsistenteDataset()
dataset_train.load_datasetSinteticoAsist(dataset_asistenteSint, 'train_steak', clases_a_entrenar=["bottle","pan_steak", "steak"])
dataset_train.load_asistentedataset(dataset_hands, 'train', clases_a_entrenar=["hand"])
dataset_train.prepare()

# Validation dataset
dataset_val = AsistenteDataset()
dataset_val.load_asistentedataset(dataset_asistente, 'val_steak', clases_a_entrenar=["bottle","pan_steak", "steak", "hand"])
dataset_val.prepare()

#Visulizamos numero de clases e imagenes en dataset de train y val:
print("TRAIN DATASET:\nImage Count: {}".format(len(dataset_train.image_ids)))
print("Class Count: {}".format(dataset_train.num_classes))
for i, info in enumerate(dataset_train.class_info):
    print("{:3}. {:50}".format(i, info['name']))
    
print("VAL DATASET:\nImage Count: {}".format(len(dataset_val.image_ids)))
print("Class Count: {}".format(dataset_val.num_classes))
for i, info in enumerate(dataset_val.class_info):
    print("{:3}. {:50}".format(i, info['name']))

#Visalizamos el numero de imagenes por clase y el numero de instancias de cada clase:
print("\n--------------------INFORMACION DE DATASET ENTRENAMIENTO------------------------: ")
dataset_train.datasetCompletoInfo()
    
print("\n--------------------INFORMACION DE DATASET VALIDACION---------------------------: ")
dataset_val.datasetCompletoInfo()


#----------------------------------------------------------------------------------------------------------
# Creamos un fichero txt en la carpeta correspondiente dentro de logs con la informacion del entrenamiento
# ---------------------------------------------------------------------------------------------------------
#Si no existe la carpeta trainnings_info en logs la creamos
trainnings_info_path=os.path.join(logs_dir, 'trainnings_info')
if(not os.path.exists(trainnings_info_path)):
    os.mkdir(trainnings_info_path)

path_txt_file=os.path.join(trainnings_info_path, 'train_info.txt')

path_txt_file=save_train_information(path_txt_file, type_of_info='config_info', object_info=config, file_new=True)
path_txt_file=save_train_information(path_txt_file, type_of_info='dataset_info', object_info=[dataset_train, dataset_val], file_new=False)

#%% 
############################################################
#  Entrenamiento del modelo
############################################################

# *** This training schedule is an example. Update to your needs ***


#----------------------------------------------------
#Definicion de Data Augmentation con libreria imgaug:
# ---------------------------------------------------
# Image Augmentation
    
#Ejemplo:
#augmentation = iaa.Fliplr(0.5) # Right/Left flip 50% of the time (solo hace flip horizontal el 50% de las veces)


#Con iaa.Sequential hacemos que se ejecuten varios procesos seguidos de DataAugmentation:
augmentation=iaa.Sequential([ 
            
        
            #1) 
            #Con iaa.Sometimes(): haces que solo se ejecute los procesos de DataAug indicados un porcentaje de las veces (Es decir, no se ejecutaran siempre)
            iaa.Sometimes( 0.2,  #Ocurrira el 30% de las veces
                          iaa.Crop(px=(0, 50))# crop images from each side by 0 to 50px (randomly chosen)
                          ),
            
            #2)
            iaa.Fliplr(0.5), # horizontally flip 50% of the images
            
            #3)
            iaa.Flipud(0.25), # vertically flip 25% of the images
            
            #4) 
            iaa.Sometimes( 0.1,  #Ocurrira el 40% de las veces
                          
                          #Con iaa.OneOf haces que se escoja aleatoriamente UNO SOLO de los procesos de DataAug indicados:
                          iaa.OneOf([ iaa.GaussianBlur(sigma=(0.5, 2.0)),  #le aplicara una sigma entre 0.5 y 2.0 segun distrib uniforme
                                     iaa.MotionBlur(k=(5,30), angle=(0,360)) #Efecto de movimiento con un kernel de entre 5 y 30 y un angulo de mov de entre 0 y 360
                                    ])
                          ),
            
            #5)
            iaa.Sometimes( 0.1,  #Ocurrira el 50% de las veces
                          
#                          iaa.OneOf([ iaa.ContrastNormalization((3, 4)), #Cambia el contraste en un factor entre 0.75 y 1.2
#                                      iaa.SigmoidContrast(gain=10, cutoff=(0,0.5)) #Efecto de movimiento con un kernel de entre 5 y 30 y un angulo de mov de entre 0 y 360
#                                    ])
                            iaa.SigmoidContrast(gain=10, cutoff=(0,0.75)) #Efecto de movimiento con un kernel de entre 5 y 30 y un angulo de mov de entre 0 y 360
                          ),
            
            #6)
            iaa.Sometimes( 0.2,  #Ocurrira el 30% de las veces
                          
                          #iaa.Affine realiza el traslacion,volteado, afinamiento, recorte, zoom, etc de las fotos 
                          iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, 
                                     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                                     rotate=(-90, 90),
                                     shear=(-8, 8) 
                                    )
                          )
                          
            ], random_order=True) # random_order=True para que aplique los procesos de manera aleatoria (no secuencial segun el orden en el que se han definido)  

                          
#----------------------------
# PROCESO DE ENTRENAMIENTO:
# ---------------------------  

stages_a_entrenar=[1,2,3]   

#layers puede ser: layers='heads',
#                         'all' 
#                         '3+'
#                         '4+'
#                         '5+'
    
        
if(1 in stages_a_entrenar):
    #------------------------------------------------
    #Entrenamiento de unicamente las ultimas capas:
    # -----------------------------------------------
    # Training - Stage 1
    tic_stage1=time.time()
    print("Training network heads")
    
    #Config of trainning stage:
    layers='heads'
    epochs_to_train=100
    learning_rate=config.LEARNING_RATE
    
    #Save stage config in txt
    path_txt_file=save_train_information(path_txt_file, type_of_info='stages_train_info', object_info=[layers, epochs_to_train, learning_rate], file_new=False)

    #Start to train:    
    model.train(dataset_train, dataset_val,
                learning_rate=learning_rate,
                epochs=model.epoch+epochs_to_train,
                layers=layers,
                augmentation=augmentation)
    
    toc_stage1=time.time()
else:
    tic_stage1=0
    toc_stage1=0


if(2 in stages_a_entrenar):
    #--------------------------------------------------------------------
    #Entrenamiento de las capas desde la 4a capa de Resnet en adelante:
    # -------------------------------------------------------------------
    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    tic_stage2=time.time()
    print("Fine tune Resnet stage 4 and up")
    
    #Config of trainning stage:
    layers='4+'
    epochs_to_train=100
    learning_rate=config.LEARNING_RATE/100
    
    #Save stage config in txt
    path_txt_file=save_train_information(path_txt_file, type_of_info='stages_train_info', object_info=[layers, epochs_to_train, learning_rate], file_new=False) 
        
    #Start to train:
    model.train(dataset_train, dataset_val,
                learning_rate=learning_rate,
                epochs=model.epoch+epochs_to_train,
                layers=layers,
                augmentation=augmentation)
    
    toc_stage2=time.time()
    
else:
    tic_stage2=0
    toc_stage2=0

if(3 in stages_a_entrenar): #PARA ESTE ENRENAMIENTO PONER 1 IMAGEN POR GPU YA QUE SINO DESBORDA LA MEMORIA
    
    #---------------------------------------------
    #Entrenamiento de TODAS las capas de la red:
    # --------------------------------------------
    # Training - Stage 3
    # Fine tune all layers
    tic_stage3=time.time()
    print("Fine tune all layers")
    
    #Config of trainning stage:
    layers='all'
    epochs_to_train=100
    learning_rate=config.LEARNING_RATE/100
    
    #Save stage config in txt
    path_txt_file=save_train_information(path_txt_file, type_of_info='stages_train_info', object_info=[layers, epochs_to_train, learning_rate], file_new=False)
        
    #Start to train:
    model.train(dataset_train, dataset_val,
                learning_rate=learning_rate,
                epochs=model.epoch+epochs_to_train,
                layers=layers,
                augmentation=augmentation)
    
    toc_stage3=time.time()
    
else:
    tic_stage3=0
    toc_stage3=0
    
#---------------------------------------------
#Visualizacion de tiempos de entrenamiento:
# --------------------------------------------
import datetime

print('\nTiempo de entrenamiento Stage1: ',str(datetime.timedelta(seconds=toc_stage1-tic_stage1))) 
print('\nTiempo de entrenamiento Stage2: ',str(datetime.timedelta(seconds=toc_stage2-tic_stage2)))
print('\nTiempo de entrenamiento Stage3: ',str(datetime.timedelta(seconds=toc_stage3-tic_stage3)))



#-------------------------------------------------------------------------------------------------------------------------------------------------
#Movemos los archivos de informacion guardada sobre la configuracion de los entrenamientos a la carpeta donde se han guardado los pesos entrenados
#-------------------------------------------------------------------------------------------------------------------------------------------------
model = modellib.MaskRCNN(mode="training", config=config, #solo lo creamos para buscar automaticamente el ultimo modelo guardado
                          model_dir=logs_dir)
path_weights_trained="\\".join(model.find_last().split('\\')[:-1])
for file_i in os.listdir(trainnings_info_path):
   
    path_txt_orig=os.path.join(trainnings_info_path,file_i)
    path_txt_dest=os.path.join(path_weights_trained,file_i)
    
    #En caso de que en la ruta destino ya exista ese nombre se busca uno que no este:
    num=0
    path_txt_dest2=path_txt_dest[:]
    while(os.path.exists(path_txt_dest2)):
       num=num+1 
       path_txt_dest2= '../..'+path_txt_dest.split('.')[-2]+('_{}.txt'.format(num))
       
    path_txt_dest=path_txt_dest2[:]
    
    #Se mueve el txt a la carpeta donde estan los pesos entrenados
    os.rename(path_txt_orig, path_txt_dest)
    

    
    
