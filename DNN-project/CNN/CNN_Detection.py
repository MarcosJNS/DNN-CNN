# -*- coding: utf-8 -*-
"""
This Scrip uses the RCNN to analyze videos captures with a camera RealSense D415. The videos are saved in .bag 
files then the CNN recognizes the objets to track them. the tracking is saved in a CSV file to feed the other DNN (FNN)
in order to infer actions.

In this DEMO we are cooking a steak. First we place the pan on the hob. Right after we put the steak with the oil and the pan.
The functions written here are a first approach to the actions problem, using the masks to fulfill conditions and 
HSV segmentation inside the functions. 

Nevertheless, the relevant part is using the tracking to infer a wider range of actions in the futur as we expand the project.

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
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import math
#from samples.asistente  
import CNN_Lib
np.set_printoptions(threshold=np.inf)
# Root directory of the project
ROOT_DIR = os.path.abspath("../")

sys.path.append(ROOT_DIR)  # To find local version of the library
directory = os.path.join(ROOT_DIR,"DNN")
print(directory)
sys.path.append(directory)
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file (to load a pre-trained weights with COCO dataset)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
   

# Import Mask RCNN
from mrcnn import visualize

import mrcnn.model as modellib

import DNNAnalysis_Lib

#-------------------
#Load configuration:
# ------------------

class AsistenteCocoInferenceConfig(CNN_Lib.CNN_Config):
    """Configuration for validation on the val dataset.
    Derives from the AsistenteConfig class and overrides some values.
    """
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    DETECTION_MIN_CONFIDENCE = 0.8 

    NUM_CLASSES = 1 + 4  # Background 
    
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
#model_path = os.path.join(ROOT_DIR, "samples/asistente/1Modelos_entrenados/18steak_plateBG/mask_rcnn_asistente_0698.h5")
model_path = os.path.join(ROOT_DIR, "CNN/Trained_Models/Steak/mask_rcnn_asistente_0698.h5")
# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

class_names=["BG", "bottle","pan_steak", "steak", "hand"]


def sarten_colocada(mask_vitro, mask_pan):
    
    intersection=cv2.bitwise_and(mask_vitro,mask_pan)
    
    area_intersection=len(np.where(intersection==255)[0])
    area_pan=len(np.where(mask_pan==255)[0])
    overlap=area_intersection/area_pan
    
    overlap_threshold=0.7 #tiene que estar 70% superpuesta la sorten sobre la vitro para considerar que se ha colocado la sarten
    if(overlap>overlap_threshold):
        return True
    else:
        return False

def centroid( mask_pan) :
    global xS1, yS1, count_displacement, vitro_mask, Ok
    
    #if sarten_colocada(vitro_mask, mask_pan) == True:
    if count_displacement == 10 :

        contours, _ = cv2.findContours(mask_pan ,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        cnts = max(contour_sizes, key=lambda x: x[0])[1]
        
            # compute the center of the contour
        M = cv2.moments(cnts)
        if M["m00"]>0:
           
            c0X = int(M["m10"] / M["m00"])
            c0Y = int(M["m01"] / M["m00"])
            xS1 = c0X
            yS1 = c0Y
            print(yS1)
            # draw the contour and center of the shape on the image
            cv2.drawContours(img2, [cnts], -1, (0, 255, 0), 2)
            cv2.circle(img2, (c0X, c0Y), 7, (255, 255, 255), -1)
            cv2.putText(img2, "center", (c0X - 20, c0Y - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        count_displacement=0
        Ok=0
        print('move ref')
                
    else:
         count_displacement += 1
             
def tracking (mask_pan):
    global tracking_vector,m
    
   # if sarten_colocada(vitro_mask, mask_pan) == True:

    _, contours, _ = cv2.findContours(mask_pan ,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    cnts = max(contour_sizes, key=lambda x: x[0])[1]
    M = cv2.moments(cnts)
    if M["m00"]>0:
       
        c0X = int(M["m10"] / M["m00"])
        c0Y = int(M["m01"] / M["m00"])
        xS1 = c0X
        yS1 = c0Y
        
        vector=xS1, yS1
        print (vector)
        tracking_vector.append(vector)

        m=m+1
    else:
        tracking_vector.append(0,0)
        m=m+1           

def saltear( mask_det):
    global xS1, yS1, img2, vitro_mask, Ok
    print(xS1,yS1)
    dis_thresholdmin=10
    dis_thresholdmax=60
    centroid(mask_det)

    displacementX=0
    displacementY=0
    contours, _ = cv2.findContours(mask_det ,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]

    if xS1 != 0 or yS1 != 0:    
        
        _, contours, _ = cv2.findContours(mask_det ,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        
        cnts = max(contour_sizes, key=lambda x: x[0])[1]
        
        M = cv2.moments(cnts)
        if M["m00"]>0:
        
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            xS2=cX
            yS2=cY
            # draw the contour and center of the shape on the image
            cv2.drawContours(img2, [cnts], -1, (0, 255, 0), 2)
            cv2.circle(img2, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(img2, "center", (cX - 20, cY - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # show the image
#                cv2.imshow("Image", img2)
#                cv2.waitKey(0)
            displacementX=np.abs(xS2-xS1)
            displacementY=np.abs(yS2-yS1)
            print('displacementX',displacementX)
            print('displacementY',displacementY)
            
            print('Ok=',Ok)
          
    if(displacementX>dis_thresholdmin) or (displacementY>dis_thresholdmin)  :
      if   (displacementX<dis_thresholdmax) and (displacementY<dis_thresholdmax)  :
        Ok+=1        
        if  Ok>=3 and paso >=3:
            #incluir salteado

            #cv2.putText(img2, 'SALTEAR', (120, 120),cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2) 
            print('salteando')
            Ok=0
            return True
        else:
            return False
    else:    
        Ok=0
        return False

def filete_colocado(mask_steak, mask_pan):

    intersection=cv2.bitwise_and(mask_pan,mask_steak)
    
    area_intersection=len(np.where(intersection==255)[0])
    area_steak=len(np.where(mask_steak==255)[0])
    overlap=area_intersection/area_steak
    print(overlap)
    overlap_threshold=0.2 #El steak ocupará un 20%de la máscara de la sartén
    if(overlap>overlap_threshold):
        return True
    else:
        return False
        
def vertiendo_aceite(mask_bottle, mask_pan, find_steak):
    global mask_steak
    
    intersection=cv2.bitwise_and(mask_pan,mask_bottle)
    
    area_intersection=len(np.where(intersection==255)[0])
    area_oil=len(np.where(mask_det==255)[0])
    overlap=area_intersection/area_oil
    
    overlap_threshold=0.2 #tiene que estar 20% superpuesta la sorten sobre la vitro para considerar que se ha colocado la sarten
    if(overlap>overlap_threshold):
        return True
    if find_steak==True:
        print('hay_filete')
        if filete_colocado(mask_steak, mask_pan):
            return True
        else:
            return False
    else:
        return False
    

    
    
    
    
#%%     
#################################
# TEST with a video
#################################

#Parametros:
#----------

video_folder='CNN_Dataset/Action_Tracking'

if(os.path.exists(os.path.join(video_folder, 'detecciones')) ==False):
    os.mkdir(os.path.join(video_folder, 'detecciones'))

list_videos=os.listdir(video_folder)
list_videos=[video for video in list_videos if(os.path.isfile(os.path.join(video_folder, video))==True)]

#list_videos=['20190711_112704.bag'] #Para ocger solo el video indicado


for video_i in list_videos[1:]:
    #Inicializamos valores del centroide
    xS1=0
    yS1=0
    count_displacement=0
    Ok=0
    tracking_vector=[]
    m=0
    K=0
    #indicamos si se quiere guardar el video o no:
    guardar_video=True
    
    #creamos un vector de colores para cada tipo de clase de objeto
    colors = visualize.random_colors(len(class_names)) #Para visualizar cada clase con un color distinto
   
#    size_video= (1920, 1080)
    size_video= (1200,720)
    rotate_video=False
#    if(video_i in videos_a_rotar):
#        size_video=(size_video[1], size_video[0])
#        rotate_video=True
    
    if(size_video[0]>1500 or size_video[1]>1500):
        size_video= (int(size_video[0]/2) , int(size_video[1]/2))
  
    if(guardar_video==True):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size_video_RGBD=(int(size_video[0]*2), size_video[1])
        cv_writer = cv2.VideoWriter(video_folder+'/detecciones/'+video_i.split('.')[0]+'_deteccion.avi', fourcc, 15.0, size_video_RGBD)
    
    cv2.namedWindow('RGBD',cv2.WINDOW_NORMAL)
#    cv2.resizeWindow('RGBD', 1920,540)
    cv2.resizeWindow('RGBD', 1200,360)  
    
    n_frame=0
    key=None
    
    #Inicializamos los objetos necesarios de la libreria pyrealsense para poder leer los videos de profundidad:
         # Create pipeline
    pipeline = rs.pipeline()
     
         # Create a config object
    config = rs.config()
         # Tell config that we will use a recorded device from filem to be used by the pipeline through playback.
    path_video_bag=os.path.join(video_folder, video_i)
    rs.config.enable_device_from_file(config, path_video_bag,   repeat_playback=False ) #IMPORTANTE PONER EL ARGUMENTO DE repeat_playback A FALSE PARA PODER DETECTAR EL FINAL DEL VIDEO
    
    # Start streaming from file
    profile = pipeline.start(config)
    
    #Desabilitamos el real time para que nos de los frames grabados uno a uno y no como si se estuviera grabando de camara
    #ya que sino se salta alguno de los frames grabados en el video debido al tiempo de procsamiento:
    playback=profile.get_device().as_playback()
    playback.set_real_time(False)
    
    #Como RGB y Depth no tienen el mismo tamanno habra que alinearlas
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    #Creamos un vector del tamaño del video en el que se almacenaran las imagenes de profundidad inciales para capturar la profundidad promedio del 
    #fondo durante los primeros frames:
    num_frames_BG=10
    images_BG_R=np.zeros([size_video[1],size_video[0],num_frames_BG])
    images_BG_G=np.zeros([size_video[1],size_video[0],num_frames_BG])
    images_BG_B=np.zeros([size_video[1],size_video[0],num_frames_BG])
    BG_mean=None
    
    #Paso de receta:
    paso=1
    framesSeguidos_aceite=0
    vertidos_de_aceite=0

    while(1):
    
        t_ini=time.time()
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        
#        Align the depth frame to color frame
        aligned_frames = align.process(frames)
        
        #Extract RGB and depth frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        #Recortamos la imagen a 1200, 360 ya que la camara de profundidad saca unas medidas mal a la izquierda:
        depth_image=depth_image[:,80:]
        color_image=color_image[:,80:]
        
        if(color_image is None):
            print('Final del video')
            break
        
        n_frame=n_frame+1
        
        if(rotate_video==True):
            color_image=imutils.rotate_bound(color_image, 90)
            depth_image=imutils.rotate_bound(depth_image, 90)
            
        #Reducimos el tamanno de imagen a la mitad para que le cueste menos tiempo el procesarla y visualizarla:
        if(np.shape(color_image)[1]>1500 or np.shape(color_image)[0]>1500):
            color_image=cv2.resize(color_image, size_video, interpolation=cv2.INTER_LANCZOS4)
            depth_image=cv2.resize(depth_image, size_video, interpolation=cv2.INTER_LANCZOS4)
        
        #Escalamos la imagen de profundidad:
        mediana=1774.0
        media= 1902.678516228748
        depth_scaled=cv2.convertScaleAbs(depth_image, alpha=255/media)
        depth_medianFiltered=cv2.medianBlur(depth_scaled, 5)
        
        #Guardamos la imagen de profundidad de los primeros frames para obtener la imagen de profundidad promedio del fondo:
        if(n_frame<num_frames_BG): #(n_frame-1) por que empieza a contar desde el 1
#            depth_BG_firstFrames[:,:,n_frame-1]=depth_medianFiltered
            images_BG_R[:,:,n_frame-1]=color_image[:,:,0]
            images_BG_G[:,:,n_frame-1]=color_image[:,:,1]
            images_BG_B[:,:,n_frame-1]=color_image[:,:,2]
            
        elif( (n_frame)==num_frames_BG):
            #Introducimos el ultimo frame:
            images_BG_R[:,:,n_frame-1]=color_image[:,:,0]
            images_BG_G[:,:,n_frame-1]=color_image[:,:,1]
            images_BG_B[:,:,n_frame-1]=color_image[:,:,2]
            
            #Calculamos la media de entre todas las fotos para cada canal y creamos la imagen RGB promedio de todas ellas:
            images_BG_R_mean=np.mean(images_BG_R, axis=-1)
            images_BG_G_mean=np.mean(images_BG_G, axis=-1)
            images_BG_B_mean=np.mean(images_BG_B, axis=-1)
            BG_mean=(np.stack([images_BG_R_mean, images_BG_G_mean, images_BG_B_mean], axis=-1)).astype('uint8')
            
            
            #Pasamos a HSV:
            frame_HSV = cv2.cvtColor(BG_mean, cv2.COLOR_BGR2HSV)
            low_H, low_S, low_V=(0,0,0)
            high_H, high_S, high_V=(180,255,46) #sacado con calibracion con trackbars
            vitro_mask = cv2.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
            
            #Erosionamos/dilatamos imagen para eliminar posible ruido:
            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            vitro_mask = cv2.morphologyEx(vitro_mask, cv2.MORPH_OPEN, kernel, iterations = 3)
            
            #Nos quedamos unicamente con el contorno mas grande:
            contours, _ = cv2.findContours(vitro_mask ,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
            biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
            vitro_mask = np.zeros(vitro_mask.shape, dtype="uint8")
            cv2.drawContours(vitro_mask, [biggest_contour], -1, (255), thickness =-1)
#            cv2.imwrite('vitro_mask.png', vitro_mask)
#            cv2.imwrite('frame_HSV.png', frame_HSV)
#            cv2.imwrite('BG_mean.png', BG_mean)
        
        #SE PROCEDE A LA DETECCION SOBRE LA IMAGEN ESCOGIDA
        t1=time.time()
        results = model.detect([color_image], verbose=0)
       
        t2=time.time()
        r = results[0] #Como solo le hemos pasado una imagen, solo cogemos el resultado 0
  
        #SE ACTUALIZAN LOS TRACKS:
        class_detections=[class_names[id] for id in r['class_ids']]
        
        #Se pasa a BGR la imagen en RGB
        color_image_BGR=cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        
        #Si ha habido alguna deteccion se pinta:
        if(len(r['class_ids'])!=0):
    #        #SE VISUALIZA EL RESULTADO DE DETECCION EN OTRA IMAGEN OBTENIDA Y SE MUESTRA DICHA IMAGEN (SOLO SE VISUALIZAN LOS OBJETOS TRACKEADOS)
           img2=CNN_Lib.display_instances(color_image_BGR, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], colors=colors) 
           
        else:
            
            img2=color_image_BGR.copy()
        
        
        #Comprobamos de todas las detecciones cuales son sartenes y comprobamos si estan colocadas sobre la 
        #vitroceramica:
        num_pans_colocated=0
        
        
        for i in np.arange(len(class_detections)):
            
            if(class_detections[i]=='pan_steak'):
                 mask_det=(r['masks'][:,:,i]).astype('uint8')*255
                 DNNAnalysis_Lib.centroid_action(mask_det,video_i, 'pan', img2,tracking_vector)
            if(class_detections[i]=='pan_steak' and paso==1):
                print(class_detections)
                mask_det=(r['masks'][:,:,i]).astype('uint8')*255
                sarten_en_vitro=sarten_colocada(vitro_mask, mask_det)
                
                if(sarten_en_vitro):
                    num_pans_colocated=num_pans_colocated+1
                
                    if(num_pans_colocated>=1):
                        paso=2
            #Comprobamos cuando se vierte aceite en caso de encontrar una botella
            elif(class_detections[i]=='bottle' and paso==2 ):
                for j in np.arange(len(class_detections)):
                    if (class_detections[j]=='pan_steak'):
                        mask_bottle=(r['masks'][:,:,i]).astype('uint8')*255
                        mask_pan=(r['masks'][:,:,j]).astype('uint8')*255
                        
                        #Si hay ya un filete se asumirá que se ha hechado previamente el aceite
                        for k in np.arange(len(class_detections)):
                            if (class_detections[k]=='steak'):  
                                find_steak=True
                                mask_steak=(r['masks'][:,:,k]).astype('uint8')*255
                                #En caso de que haya alguna sarten en la vitro se comprueba si se esta vertiendo aceite en alguna:
                            else:
                                find_steak=False
                            aceite_vertido=vertiendo_aceite(mask_bottle, mask_pan,find_steak)
                            if(aceite_vertido):
                                print('vertiendo aceite')
                                framesSeguidos_aceite=framesSeguidos_aceite+1
                                paso=3
                            elif(framesSeguidos_aceite>0): #solo entrara aqui si no esta vertiendo aceite
                                vertidos_de_aceite=vertidos_de_aceite+1
                                framesSeguidos_aceite=0
                                      
                if(vertidos_de_aceite>=1):
                    paso=3
                    

                 
            if(class_detections[i]=='steak' and paso==3 ):
                for j in np.arange(len(class_detections)):
                
                    if (class_detections[j]=='pan_steak'):
                        mask_steak=(r['masks'][:,:,i]).astype('uint8')*255
                        mask_pan=(r['masks'][:,:,j]).astype('uint8')*255
                        
                        if(filete_colocado(mask_steak, mask_pan)==True):
                            paso=5
            if class_detections[i]=='pan_steak' and paso>=5:
               
               mask_pan=(r['masks'][:,:,i]).astype('uint8')*255
               centroid(mask_pan) 
               saltear(mask_pan)                            
               if saltear(mask_pan) == True :
                   paso=6
            if class_detections[i]=='pan_steak' and paso==6 :
                mask_pan=(r['masks'][:,:,i]).astype('uint8')*255
                tracking(mask_pan)
                for j in range(len(tracking_vector)):
                    couple=tracking_vector[j]
                    cv2.circle(img2, (couple[0], couple[1]), 7, (255, 255, 255), -1)   
                            
        offset=35
        
        
        Txt1='Receta:Freir Filete'
        Txt2='Poner Sarten '
        Txt3='Echar Aceite '
        Txt4='Calentar Aceite '
        Txt5='Poner Filete '
        Txt6='Saltear'
        Txt7='Listo para comer'
        a=255;b=255;c=255;d=255;e=255;f=255;
        if(paso==1): a=0
        if(paso==2): b=0   
        if(paso==3): c=0
        if(paso==4): d=0
        if(paso==5): e=0 
        if(paso==6): f=0   
        
        
        cv2.putText(img2,Txt1,(10,25),cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        
        cv2.putText(img2,Txt2,(10,60),cv2.FONT_HERSHEY_COMPLEX, 1, (a, 255, a), 2)
        
        cv2.putText(img2,Txt3,(10,95),cv2.FONT_HERSHEY_COMPLEX, 1, (b, 255, b), 2)
        
        cv2.putText(img2,Txt4,(10,95+offset),cv2.FONT_HERSHEY_COMPLEX, 1, (c, 255, c), 2)
        
        cv2.putText(img2,Txt5,(10,130+offset),cv2.FONT_HERSHEY_COMPLEX, 1, (d, 255, d), 2)
        
        cv2.putText(img2,Txt6,(10,165+offset),cv2.FONT_HERSHEY_COMPLEX, 1, (e, 255, e), 2)
        
        cv2.putText(img2,Txt7,(10,200+offset),cv2.FONT_HERSHEY_COMPLEX, 1, (f, 255, f), 2)
       
           
        
        #SE GUARDA EL FRAME ACTUAL EN EL VIDEO :
        if(guardar_video):
            size = img2.shape[1], img2.shape[0]
            
            if size[0] != size_video[0] or size[1] != size_video[1]:
                img2 = cv2.resize(img2, size_video, interpolation=cv2.INTER_LANCZOS4)
                
            #juntamos la imagen RGB de deteccion y la de profundidad:
            depth_medianFiltered_RGB=cv2.cvtColor(depth_medianFiltered, cv2.COLOR_GRAY2BGR)
            img_RGBD=np.hstack((img2, depth_medianFiltered_RGB))
            cv_writer.write(img_RGBD)
       
        cv2.imshow('RGBD',img_RGBD)
        
        t_fin=time.time()
        
        #SE SACA POR PANTALLA EL TIEMPO QUE LE HA COSTADO LA DETECCION, Y EL TIEMPO DE DETECCION+VISUALIZACION:
        if(n_frame==27):
             pass
        print('\nFrame: ',n_frame)
        print('Tiempo de deteccion:', t2-t1)
        print('Tiempo de deteccion+visualizacion:', t_fin-t_ini)  
    
        #SI HA ACABADO EL VIDEO SE SALE DEL BUCLE:
        if(playback.current_status() == rs.playback_status.stopped):
             print('Final del video')
             break
        
        #SI SE PRETA esc O space SE SALE DEL BUCLE:
        key= cv2.waitKey(1)
        
        if key == ord(' ') or key == 27: #Pretar 'espacio' para pasar al siguiente video
           
            while(playback.current_status() != rs.playback_status.stopped):
                 frames = pipeline.try_wait_for_frames()
            
            break
              
        
    if(guardar_video):
        cv_writer.release()

    cv2.destroyAllWindows()
#    pipeline.stop()
    if(n_frame == 0):
        raise NameError('No se ha encontrado el video')
        
    if( key == 27): #Si se preta escape no sigue con el siguiente video
        break

          


     
     