# -*- coding: utf-8 -*-
"""
Script utilizado para la ejecucion de las red Mask RCNN sobre un video. Para ello se le cargan los pesos
entrenados y se guardan los resultados obtenidos del video.

"""

############################################################
#  CONSTRUCCION DEL MODELO
############################################################
import os
import sys
import numpy as np
import time
import cv2
import imutils 

#from centroidtracker import CentroidTracker #Clase creada para llevar a cabo el seguimiento de objetos encontrados en el video
import sort.sort_asis as sort

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

sys.path.append(ROOT_DIR)  # To find local version of the library

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file (to load a pre-trained weights with COCO dataset)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
   

# Import Mask RCNN
from mrcnn import visualize

import mrcnn.model as modellib

from samples.asistente import asistente


#-------------------
#Load configuration:
# ------------------
config = asistente.AsistenteConfig()
config.display()

#------------------
# Inference config:
# -----------------   

class AsistenteCocoInferenceConfig(asistente.AsistenteConfig):
    """Configuration for validation on the val dataset.
    Derives from the AsistenteConfig class and overrides some values.
    """
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    DETECTION_MIN_CONFIDENCE = 0.8 

    NUM_CLASSES = 1 + 6  # Background 
    
inference_AsistenteConfig = AsistenteCocoInferenceConfig()

#---------------
# Create model:
# --------------

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_AsistenteConfig,
                          model_dir=MODEL_DIR)

#------------------------------------
# Load weights of trained model:
# -------------- ---------------------

# Get path to saved weights
# Either set a specific path or find last trained weights
model_path = os.path.join(ROOT_DIR, "samples/asistente/1Modelos_entrenados/12Sintetico_NoAleat_manosSinPersonas/mask_rcnn_asistente_0119.h5")

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

class_names=["BG","bottle","pan", "pot","knife", "potato", "hand"]

#%%     
#################################
# TEST with a video
#################################

#Parametros:
#----------
videos_a_rotar=['Video0.mp4',
                'IMG_5092.MOV']

video_folder='asistente_dataset/test_videos4'

if(os.path.exists(os.path.join(video_folder, 'detecciones')) ==False):
    os.mkdir(os.path.join(video_folder, 'detecciones'))

list_videos=os.listdir(video_folder)
list_videos=[video for video in list_videos if(os.path.isfile(os.path.join(video_folder, video))==True)]
for video_i in list_videos:
    
    
    video_path=os.path.join(video_folder, video_i)

    guardar_video=True
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    colors = visualize.random_colors(len(class_names)) #Para visualizar cada clase con un color distinto
   
    size_video= (int(cap.get(3)) , int(cap.get(4)))
    rotate_video=False
    if(video_i in videos_a_rotar):
        size_video=(size_video[1], size_video[0])
        rotate_video=True
    
    if(size_video[0]>1500 or size_video[1]>1500):
        size_video= (int(size_video[0]/2) , int(size_video[1]/2))
  
    if(guardar_video==True):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
        cv_writer = cv2.VideoWriter(video_folder+'/detecciones/'+video_i.split('.')[0]+'_deteccion.avi', fourcc, fps, size_video)
    
    n_frame=0
    
    #Inicializamos el tracker de objetos vistos:
#    centroidTracker=CentroidTracker(maxDisappeared=6)
    tracker = sort.Sort(max_age=8 ,min_hits=5) 
    
    key=None
    
    while(cap.isOpened()):
    
            
        ret, frame = cap.read()
        
        if(frame is None):
            print('Final del video')
            break
        
        n_frame=n_frame+1
        
        if(rotate_video==True):
            frame=imutils.rotate_bound(frame, 90)
            
        #Reducimos el tamanno de imagen a la mitad para que le cueste menos tiempo el procesarla y visualizarla:
        if(np.shape(frame)[1]>1500 or np.shape(frame)[0]>1500):
            frame=cv2.resize(frame, size_video, interpolation=cv2.INTER_LANCZOS4)
        
        #SE PROCEDE A LA DETECCION SOBRE LA IMAGEN ESCOGIDA
        frame_RGB=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t1=time.time()
        results = model.detect([frame_RGB], verbose=0)
        t2=time.time()
        r = results[0] #Como solo le hemos pasado una imagen, solo cogemos el resultado 0
        
        #SE ACTUALIZAN LOS TRACKS:
        class_detections=[class_names[id] for id in r['class_ids']]
        tracked_objects, metainfo_tracks = tracker.update(r['rois'], class_detections)
        
        #SE PINTAN LOS CENTROIDES E ID DE LOS TRACKS OBTENIDOS CON "SORT" TRACKER:
        for idx_track in np.arange(tracked_objects.shape[0]):
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            startX, startY, endX, endY= (tracked_objects[idx_track,0:4]).astype('int32')
            centroid=(int((startX + endX) / 2.0), int((startY + endY) / 2.0))

            text = "ID {}".format(int(tracked_objects[idx_track,4]))
            cv2.putText(frame, text, (centroid[1] - 10, centroid[0] - 10),	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(frame, (centroid[1], centroid[0]), 4, (0, 0, 255), -1)
            cv2.putText(frame, metainfo_tracks['class_of_tracks'][idx_track], (centroid[1] + 10, centroid[0]+ 4),	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, metainfo_tracks['movState_of_tracks'][idx_track], (centroid[1] + 10, centroid[0]+ 15),	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            #Si el track ha sido solo predicho y no detectado, se pinta tambien la boundingbox predicha
            if(metainfo_tracks['traks_onlyPred'][idx_track]==True):
                cv2.rectangle(frame, (startY, startX), (endY, endX), color=(0, 0, 255), thickness=4)
        
        #SE VISUALIZA EL RESULTADO DE DETECCION EN OTRA IMAGEN OBTENIDA Y SE MUESTRA DICHA IMAGEN
        img2=asistente.display_instances(frame, r['rois'], r['masks'], r['class_ids'], 
                                          class_names, r['scores'], colors=colors)
        
        #SE GUARDA EL FRAME ACTUAL EN EL VIDEO :
        if(guardar_video):
            size = img2.shape[1], img2.shape[0]
            
            if size[0] != size_video[0] or size[1] != size_video[1]:
                img2 = cv2.resize(img2, size_video)
                
            cv_writer.write(img2)
        
        cv2.imshow('frame',img2)
        
        t3=time.time()
        
        #SE SACA POR PANTALLA EL TIEMPO QUE LE HA COSTADO LA DETECCION, Y EL TIEMPO DE DETECCION+VISUALIZACION:
        print('\nFrame: ',n_frame)
        print('Tiempo de deteccion:', t2-t1)
        print('Tiempo de deteccion+visualizacion:', t3-t1)  
    
        key= cv2.waitKey(1)
        
        if key == ord(' '): #Pretar 'espacio' para pasar al siguiente video
            
            if(guardar_video):
                cv_writer.release()
                
            cap.release()
            cv2.destroyAllWindows()
            
                
            break
        
        if key == 27: #Pretar 'esc' para salir de la reproduccion
            
            if(guardar_video):
                cv_writer.release()
                
            cap.release()
            cv2.destroyAllWindows()
            break
    
    
    if(guardar_video):
        cv_writer.release()
    cap.release()
    cv2.destroyAllWindows()
    
    if(n_frame == 0):
        raise NameError('No se ha encontrado el video')
        
    if( key == 27): #Si se preta escape no sigue con el siguiente video
        
        break

#UTILIZAMOS LA FUNCION DE DETECCION CREADA:
#detecta_video('asistente_dataset/videos_test/Video2.avi', model, guardar_video=True)
#detecta_video('Video_nuevo.avi')
