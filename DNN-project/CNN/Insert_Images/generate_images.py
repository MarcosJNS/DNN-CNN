# -*- coding: utf-8 -*-
"""
Main scrip for dataset creation. Here, different objects and backgrounds are used to create synthetic images.
To know how to create such images just execute the code and the instructions will show up in the console.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import pickle
import json
from datetime import datetime

############################################################################################
#                               CODIGO DEL PROGRAMA PRINCIPAL
############################################################################################

def main():
    """
    MAIN PROGRAM
    
    Once you run it, initially it will appear an openCV window with the background selected in order to add the desired objects to create already tagged data.
    It is important to read the funcionalaties so you can use it confortably. You can un
    Leer las siguientes funcionalidades para poder pegar, deshacer, cambiar de objeto de una misma clase, crear nueva imagen con otro fondo, etc.
    
    
    You can do this actions with the mouse:
        
        Move the mouse: As you move through the window you will see the final placement of the object on top the background.
        
        Scroll wheel: Enlarge or shorten objects.
            
        Left button: Stick the object to the image. The mask will be recorded in the intance "resultado" from the class "imagen_resultado" that has been created.

                         
        Right button:  Undo the last step, erase the last object attached to the image.
        
    You can use the keyboard too:
        
        Press 'esc':  It closes the window and represents the results with matplotlib. The results are saved in the variable 'resultado'. q
         
        Press 'c': The object category changes to the next one, you can swith between all the categories stocked in the dataset.
        
        Press 's': Manually selection of the object category by introducing its ID trough the terminal. 
        
        Press 'space': Switch to the next object in the same class.
        
        Press 'b': It saves the picture and imports a new background to restart the process.
   
        
    
    """
     
    #Definimos e incializamos variables globales que queremos que sean leidas/escritas desde cualquier parte del codigo
    global image_BG
    global scale
    global image_obj_orig, mask_obj_orig
    global image_obj_rescaled, mask_obj_rescaled
    global dir_img_FG, dir_img_BG
    global image_new, mask_new    
    global resultado, resultado_ant
    
    #Indicamos la ruta donde queremos almacenar las imagenes creadas: 'train' o 'val 
    images_created_path='train_steak' #'train' o 'val'
    
    #Si no exite la carpeta indicada se crea una con ese nombre para almacenar las imagenes creadas:
    if(os.path.exists(images_created_path) ==False):
         os.mkdir(images_created_path)
    
    #Creamos un diccionario donde almacenaremos el resultado total con tadas las imagenes creadas:
    images_created=dict()

    #Caragamos la imagen de fondo inicial (en este caso se comienza desde el 1, pero puedes empezar desde otro mas avanzado):
    num_BG=0 #al llamar selecciona_fondo_nuevo le suma uno por lo que ponemos el que queremos -1
         
    if(images_created_path=='train'): #Dependiendo de si hemos indicado carpeta train o val coge unas imagenes de objetos u otras
        path_BGs='BG'
    elif(images_created_path=='train_steak'): #Dependiendo de si hemos indicado carpeta train o val coge unas imagenes de objetos u otras
        path_BGs='BG_steak'
    elif(images_created_path=='val'):
        path_BGs='BG_val'

    num_BG=selecciona_fondo_nuevo(path_BGs, num_BG)
    
    #Cargamos una categoria de objeto (la primera de todas inicialmente):  
    desired_categories=None #Con None se cogen todas las categorias de objetos
    if(images_created_path=='train'): #Dependiendo de si hemos indicado carpeta train o val coge unas imagenes de objetos u otras
        path_all_categories='objects'
    elif(images_created_path=='train_steak'):
        path_all_categories='objects'
        desired_categories=['bottle', 'pan_steak', 'steak', 'steak_cooked']
        
    elif(images_created_path=='val'):
        path_all_categories='objects_val'
        
        
    category_id=-1 #como al llamar a selecciona_categoria_nueva se le sumara 1 a esa ID se incializa con -1
    category_id, obj_catgory=selecciona_categoria_nueva(path_all_categories, category_id, desired_categories=desired_categories, select_in_list=False) 
    path_obj_category=os.path.join(path_all_categories, obj_catgory)
    
    #Cargamos un objeto de la categoria indicada:
    selecciona_objeto_nuevo(path_obj_category) #No devuelve nada ya que modifica variables globales
    
    #Creamos una instancia de la clase creada imagen_resultado, para almacenar los resultados obtenidos al pegar objetos
    resultado=imagen_resultado(dir_img_BG)
    resultado_ant=imagen_resultado(dir_img_BG) #Se utiliza como copia de seguridad por si quieres volver un paso atras (al principio se inicializa igual que resultado para tener el fondo como image_masked)

    #Creamos una ventana de openCV donde mostraremos la imagen e interactuaremos en ella con el raton para posicionar objetos:
    windowName = 'Imagen'
    cv2.namedWindow(windowName)
    
    #Asignamos la ventana creada a la funcion callback del raton:
    cv2.setMouseCallback(windowName, mouseCallBack)
    
    #Bucle pricipal en el que representamos la imagen obtenida, y leemos tecla presionada en teclado para realizar acciones:
    try:
        while(True):
            
            #Leemos la tecla presionada
            key = cv2.waitKey(20)
            
            #En caso de haber presiondo tecla 'esc' se sale del programa
            if key == 27:
                break
            
            #En caso de presionar 'c' pasa a pegar la siguiente categoria de objetos (segun el orden en que apaerecen en list_obj_categories)
            elif key == ord('c'): 
                category_id, obj_catgory=selecciona_categoria_nueva(path_all_categories, category_id, desired_categories=desired_categories, select_in_list=False) 
                path_obj_category=os.path.join(path_all_categories, obj_catgory)
                
                #Se actualiza el objeto a mostrar a uno de la nueva categoria
                selecciona_objeto_nuevo(path_obj_category) #No devuelve nada ya que modifica variables globales
            
            #En caso de presionar 's' se selecciona manualmente la categoria de objeto a la que se quiere cambiar introduciendo su ID por el terminal:
            elif key == ord('s'): 
                category_id, obj_catgory=selecciona_categoria_nueva(path_all_categories, category_id, desired_categories=desired_categories, select_in_list=True) 
                path_obj_category=os.path.join(path_all_categories, obj_catgory)
                
                #Se actualiza el objeto a mostrar a uno de la nueva categoria
                selecciona_objeto_nuevo(path_obj_category) 
               
            #En caso de presionar 'espacio' pasa a buscar otro objeto aleatorio de uno de los de la carpeta path_obj_category:
            elif key == ord(' '): 
                selecciona_objeto_nuevo(path_obj_category) 
            
            #En caso de presionar 'b' pasa a buscar el siguiente fondo que hay en la carpeta 'BG':
            elif key == ord('b'): 
                
                #Se guarda la imagen que se ha creado en caso de que se haya pegado alguna mascara:
                if(len(resultado.labels_FG) != 0):
                    
                    #Cogemos el timestamp para annadirselo al nombre, y asi no sobreescribir otra imagen:
                    now = datetime.now()
                    timestamp = now.strftime("%Y%m%d_%Hh%Mm%Ss")
                    
                    #Annadimos el timestamp al nombre delante para que las ultimas imagenes creadas se
                    #queden al final de la carpeta. El nombre del fondo lo leemos para asegurarnos que 
                    #coicide con el fondo que se ha puesto
                    name_BG=(dir_img_BG.split(os.sep)[-1]).split('.')[0]
                    img_name=timestamp+'_'+name_BG+'.png'
                                           
                    images_created[img_name]=resultado
                    cv2.imwrite(os.path.join(images_created_path,img_name), resultado.image_masked)
                    
                    #Representamos la imagen creada con sus mascaras:
    #                resultado.representa_resultado(3)
                    
                #Se pasa al siguiente fondo:
                num_BG=selecciona_fondo_nuevo(path_BGs, num_BG) #No devuelve nada ya que modifica variables globales
                
                #Se vuelve a inicializar el resultado para la nueva imagen:
                resultado=imagen_resultado(dir_img_BG)
                resultado_ant=imagen_resultado(dir_img_BG) 
                
            #Representamos la imagen obtenida hasta el momento en la ventana openCV:
            cv2.imshow(windowName, image_new)
            
        #Si al pretar esc se habia pegado algun objeto en la ultima imagen editada,
        #se pregunta por terminal si se quiere guardar la ultima imagen:
        if(len(resultado.labels_FG) != 0):
            print('########################################################')
            print('                       ATENCION!!                       ') 
            print('########################################################')
                  
            if(input('¿Deseas guardar la ultima imagen editada? [s/n]:  ')=='s'):
                
                #Cogemos el timestamp para annadirselo al nombre, y asi no sobreescribir otra imagen:
                now = datetime.now()
                timestamp = now.strftime("%Y%m%d_%Hh%Mm%Ss")
                
                #Annadimos el timestamp al nombre delante para que las ultimas imagenes creadas se
                #queden al final de la carpeta. El nombre del fondo lo leemos para asegurarnos que 
                #coicide con el fondo que se ha puesto
                name_BG=(dir_img_BG.split(os.sep)[-1]).split('.')[0]
                img_name=timestamp+'_'+name_BG+'.png'
                
                images_created[img_name]=resultado
                cv2.imwrite(os.path.join(images_created_path,img_name), resultado.image_masked)
                    
                #Representamos la imagen creada con sus mascaras:
    #            resultado.representa_resultado(3)
        
        #Guardamos las anotaciones de las imagenes creadas en ficheros .pkl (con mascaras incluidas) y .json (sin mascaras)
        if(len(images_created) != 0):
            
            #Se guardan en un archivo .pkl todas las anotaciones de las imagenes realizadas
            #Si ya habia en la carpeta indicada un archivo de anotaciones pkl, se añaden las anotaciones a las ya existentes 
    #        guarda_anotaciones_enPKL(images_created, images_created_path)
            
            
            #Se guardan en un archivo .json todas las anotaciones de las imagenes realizadas
            #En el fichero json no se annaden las mascaras creadas, solo se guarda la informacion necesaria
            #para volver a generar la misma imagen guardada y sus mascaras.
            #Si ya habia en la carpeta indicada un archivo de anotaciones json, se añaden las anotaciones a las ya existentes 
            guarda_anotaciones_enJSON(images_created, images_created_path)
        
        #Cerramos las ventanas de opencv
        cv2.destroyAllWindows()
    
    except:
        print('guardado JSON tras ocurrir un error' )
        
        #En caso de que salte algun error guardamos el json de las imagenes creadas hasta el momento
        #Guardamos las anotaciones de las imagenes creadas en ficheros .pkl (con mascaras incluidas) y .json (sin mascaras)
        if(len(images_created) != 0):
            
            #Se guardan en un archivo .json todas las anotaciones de las imagenes realizadas
            #En el fichero json no se annaden las mascaras creadas, solo se guarda la informacion necesaria
            #para volver a generar la misma imagen guardada y sus mascaras.
            #Si ya habia en la carpeta indicada un archivo de anotaciones json, se añaden las anotaciones a las ya existentes 
            guarda_anotaciones_enJSON(images_created, images_created_path)
        

      
##################################################################################################
#        Definimos las funciones que se usan desde el programa principal, u otras funciones
##################################################################################################

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
        
    #transformamos la mascara de 0s y 255s, a 0s y 1s
    mask_crop = mask_crop.astype(bool).astype('uint8') 
    
    return image_crop, mask_crop


def cambio_escala_obj(image_obj, mask_obj, scale=1.0):
    
    """
    Funcion que realiza un cambio de escala de una imagen de un objeto y su mascara en funcion de una proporcion indicada en % 
    respecto de la imagen original y que mantiene las proporciones de la imagen.
    
    ----------------------
    Parametros de entrada:
    ----------------------
    image_obj: imagen en la que aparece el objeto 
    mask_obj: mascara binaria del objeto que aparece en image_obj
    scale:  reducción: numero de 0 a 1 para indicar el porcentaje de reduccion de la escala del objeto.
            ampliación: numero decimal mayor que 1 para indicar el porcentaje que se desea ampliar
    
    ----------------------
    Parametros de salida:
    ----------------------   
    image_out: imagen del objeto cambiada de escala
    mask_out: mascara binaria del objeto cambiada de escala
    
    """
            
    new_height=image_obj.shape[0]*scale
    aspect_ratio = new_height / image_obj.shape[0]
    new_width=image_obj.shape[1]*aspect_ratio
            
    image_out=cv2.resize(image_obj, (int(new_width),int(new_height)), interpolation=cv2.INTER_LANCZOS4)
    mask_out=cv2.resize(mask_obj, (int(new_width),int(new_height)), interpolation=cv2.INTER_LANCZOS4)
    
    return image_out, mask_out

def ajusta_obj_a_fondo(size_img_BG, image_obj, mask_obj, size_obj_orig, scale):
    """
    Funcion que cambia la escala del objeto en caso de que la imagen del objeto fuera mas grande que la 
    imagen de fondo. En ese caso se cambia el tamanno del objeto al del fondo.
    
    ----------------------
    Parametros de entrada:
    ----------------------
    size_img_BG: dimensiones de la imagen de fondo
    image_obj: imagen del objeto a pegar
    mask_obj: mascara del objeto a pegar
    size_obj_orig: dimensiones originales de la imagen del objeto a pegar
    scale: escala a la que esta la imagen del objeto respecto de sus dimensiones originales
       
    ----------------------
    Parametros de salida:
    ----------------------
    image_obj_new: imagen del objeto obtenida (reescalada si hubiera hecho falta)
    mask_obj_new: mascara del objeto obtenida 
    scale_new: escala obtenida 
    
    """
    #En caso de que la imagen del objeto a pegar se sale de la foto de fondo:
    if( size_img_BG[0]<image_obj.shape[0] or size_img_BG[1]<image_obj.shape[1] ): 
        
        #Miramos que lado de la imagen a pegar es el más grande:
        idx_max=np.argmax(image_obj.shape[:-1])
        
        #Reescalamos la imagen a pegar para que el lado mas grande de la imagen a pegar quepa en 
        #la imagen de fondo, PERO MANTENIENDO EL ASPECT RATIO:
        if(idx_max==0): #Imagen a pegar mas alta que ancha
            
            new_height=size_img_BG[0]
            aspect_ratio = new_height / image_obj.shape[0]
            new_width=image_obj.shape[1]*aspect_ratio
            
            
        elif(idx_max==1): #Imagen a pegar mas ancha que alta
            new_width=size_img_BG[1]
            aspect_ratio = new_width / image_obj.shape[1]
            new_height=image_obj.shape[0]*aspect_ratio
        
        image_obj_new=cv2.resize(image_obj, (int(new_width),int(new_height)), interpolation=cv2.INTER_LANCZOS4)
        mask_obj_new=cv2.resize(mask_obj, (int(new_width),int(new_height)), interpolation=cv2.INTER_LANCZOS4)
        
        #Calculamos la escala a la que se a quedado la imagen respecto de la original, para partir de ella en caso de
        #querer reducir la escala con la ruleta:
        scale_new=round(image_obj_new.shape[0]/size_obj_orig[0], 2)
        
    #En caso contrario:
    else:
        image_obj_new=image_obj
        mask_obj_new=mask_obj
        scale_new=scale
    
    return image_obj_new, mask_obj_new, scale_new
 
def blending(image_BG, image_FG, mask, num_capas=3, grosor_capa=1):
    """
    Funcion que hace blending del contorno del objeto a pegar, con la imagen de fondo sobre la que 
    se pega (fusion progresiva entre el borde del FG con el BG)
    
    ----------------------
    Parametros de entrada:
    ----------------------
    image_BG: imagen de fondo
    image_FG: imagen del objeto a pegar
    mask: mascara del objeto a pegar
    num_capas: numero de erosiones (capas de pixeles) a las que se les aplica un alfa para el blending
               (si indicas 3, tendras 3 capas en las que se fusiona el objeto y el fondo)
    grosor_capa: numero de pixeles de grosor de cada capa 
       
    ----------------------
    Parametros de salida:
    ----------------------
    image_blended: imagen obtenida tras pegar el objeto sobre el fondo aplicando blending sobre el 
                   borde entre el objeto y el fondo.
    
    """
    #creamos un array numpy para guardar las distintas capas de la erosion.
    #Las capas se guardan con un valor de 0 a 1 aplicando un alpha distinto a cada una.
    #La primera capa guardada es la mascara completa:
    mask_layers=mask.copy()
    mask_layers=np.expand_dims(mask_layers, axis=-1)
    mask_layers=mask_layers.astype(bool).astype('float16') #Guadaremos las capas con un float de 0 a 1
    
    #Obtenemos el valor de los alphas a poner a cada capa:
    step=round(1/num_capas, 2)
    alphas=list(np.arange(step, 1, step=step))
    if(len(alphas)<num_capas):
        alphas.append( round(np.mean([alphas[-1],1]), 2) ) #si le falta un valor de alphas se hace la media entre el ultimo alpha y '1' para que haya el mismo numero de aphas que capas
    
    #Obtenemos cada capa y le aplicamos el alpha correspondiente:
    mask_erode=mask.copy()
    mask_erode_ant=mask.copy()
    
    for idx_capa in np.arange(num_capas):
        #Realizamos erosion de la mascara para quitar un capa
        kernel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        mask_erode = cv2.erode(mask_erode,kernel,iterations = grosor_capa)
        
        #obtenemos la capa que ha sido erosionada:
        layer_erode=mask_erode_ant-mask_erode 
        layer_erode=np.expand_dims(layer_erode, axis=-1)
        layer_erode=layer_erode.astype(bool).astype('float16')
        
        #Le aplicamos el apha correspondiente y la almacenamos:
        layer_erode=layer_erode*alphas[idx_capa]
        mask_layers=np.append(mask_layers, layer_erode, axis=-1)
        
        #Actualizamos la nueva mascara a erosionar
        mask_erode_ant=mask_erode.copy()
    
    #sustituimos la mascara incial (guardada en la primera posicion de mask_layers) por la final erosionada
    #para que las capas erosionadas esten a 0:
    mask_layers[:,:,0]=mask_erode.astype(bool).astype('float16')
    
    #Sumamos todas las capas para obtener la mascara final con la que hacer el blending:
    mask_alpha=mask_layers.sum(axis=2)
    
    #calculamos la mascara invertida:
    mask_beta=1.0-mask_alpha
    
    #Para poder multiplicar por la mascara las imagenes tienen que tener el mismo numero de dimensiones:
    mask_alpha=np.expand_dims(mask_alpha, axis=-1)
    mask_beta=np.expand_dims(mask_beta, axis=-1)  
#    plt.imshow((mask_alpha[:,:,0]*255).astype('uint8'))
#    plt.show()
    
    #Hacemos el blending:
    image_blended=(image_FG*mask_alpha) + (image_BG*mask_beta)
    image_blended=image_blended.astype('uint8')
    
    return image_blended
       
        
def aplica_mascara(image_BG, image_obj, mask_obj, pos_xy, blend_obj=False, muestra_resultado=False):
    
    """
    Funcion que realiza un pegado de objetos sobre un fondo, introduciendole la imagen de fondo, la imagen del objeto,
    la mascara del objeto, y la posicion donde se quiere pegar el objeto centrado
    
    ----------------------
    Parametros de entrada:
    ----------------------
    image_BG: imagen de fondo sobre la que se quiere pegar el objeto
    image_obj: imagen en la que aparece el objeto que se quiere pegar
    mask_obj: mascara binaria del objeto que aparece en image_obj
    pos_xy: lista de dos elementos en el que se indica la posicion en pixeles en la que se quiere pegar el objeto sobre la imagen de fondo
            utilizando el siguiente formato: pos_xy=[x, y]
    blend_obj: booleano para indicar si se quiere hacer blending (fusion progresiva entre el borde del FG con el BG) al objeto a pegar
    muestra_resultado: booleano para indicar si se quiere representar el objeto ya enmascarado
    
    ----------------------
    Parametros de salida:
    ----------------------   
    outImage: imagen resultante tras pegar el objeto en el fondo
    mask_new: mascara del objeto pegado en la nueva imagen
    
    """   
    
    #Obtenemos el objeto deseado ya segmentado con la mascara sobre un fondo negro
    foreground_obj=cv2.bitwise_and(image_obj,image_obj, mask= mask_obj)
    
    #Creamos una imagen negra que sea del mismo tamanno que la imagen de fondo que sera la mascara del objeto en la imagen final
    #Tambien necesitaremos el foreground final, por lo que habra que pegar el foreground del objeto en una imagen negra del tamanno del fondo
    mask_new=np.zeros(image_BG.shape[:-1], dtype='uint8')
    foreground_new=np.zeros(image_BG.shape, dtype='uint8')
    
    #Calculamos los puntos que definen la boundingbox del objeto en la imagen nueva
    #Hay que tener encuenta que si la imagen del objeto no tiene un numero par de pixeles, no habra un pixel justo centrado en la imagen
    offset1=(np.array(mask_obj.shape)/2).astype(int)
    dim_impares=(np.array(mask_obj.shape)%2).astype(bool)
    
        #En caso de que alguna de las dimensiones de la imagen del objeto sea impar, habra un offset distinto para x1,y1 que para x2,y2
    if(True in dim_impares ):  
        idx_impar=np.where(dim_impares == True)[0]
        
        if(len(idx_impar)==2): #si las dos dimensiones son impares, le sumaremos 1 tanto a x2 como y2 con el offset2
            offset2=offset1+1
        elif(idx_impar==0): #Si es impar el numero de filas, se le sumara 1 en esa dimension (se le sumara a y2)
            offset2=[offset1[0]+1, offset1[1]]
        else:         #Si es impar el numero de columnas, se le sumara 1 en esa dimension (se le sumara a x2)
            offset2=[offset1[0], offset1[1]+1]
    else:
        offset2=offset1
        
        #hacemos el flip de offsets ya que las 'x' corresponden con las columnas de matriz, y las 'y' con las filas    
    offset1=np.flip(offset1) 
    offset2=np.flip(offset2)
        
        #Calculamos los puntos que definen la imagen del objeto en el nuevo fondo
    x1,y1 = pos_xy-offset1
    x2,y2 = pos_xy+(offset2-1) #Hay que tener en cuenta el pixel central, por eso -1 al offset2

        
    #Hay que contemplar que x1,y1,x2,y2 esten fuera de la imagen y recortar la imagen del objeto en ese caso para poder realizar el pegado
    #de objetos en bordes de la imagen:
    if(x1<0 or y1<0 or x2>(image_BG.shape[1]-1) or y2>(image_BG.shape[0]-1)): 
    
        #Inicializamos variable que usaremos para saber cuantos pixeles se sale la imagen del objeto en la imagen de fondo
        outX_L=0 #cuantos pixeles se salen en X por la izquierda
        outX_R=0 #" " "                         por la derecha
        outY_U=0 #cuantos pixeles se salen en Y por arriba
        outY_D=0 #"  "  "                       por abajo
        
        #Comprobamos por que lado se sale la imagen del objeto, y actualizamos x1,y1,x2,y2 para que no se salgan
        if(x1<0):
            outX_L=abs(x1)
            x1=0
        
        if(y1<0):
            outY_U=abs(y1)
            y1=0
        
        if(x2>(image_BG.shape[1]-1)):
            outX_R=x2-(image_BG.shape[1]-1)
            x2=(image_BG.shape[1]-1)
            
        if(y2>(image_BG.shape[0]-1)):
            outY_D=y2-(image_BG.shape[0]-1)
            y2=(image_BG.shape[0]-1)
            
        #Recortamos la imagen, mascara, y foreground del objeto para que coincida con el pegado que se desea hacer:
        x1_obj=outX_L
        y1_obj=outY_U
        x2_obj=(mask_obj.shape[1]-1)-outX_R
        y2_obj=(mask_obj.shape[0]-1)-outY_D    
            
        image_obj=image_obj[y1_obj:y2_obj+1,x1_obj:x2_obj+1]
        mask_obj=mask_obj[y1_obj:y2_obj+1,x1_obj:x2_obj+1]
        foreground_obj=foreground_obj[y1_obj:y2_obj+1,x1_obj:x2_obj+1]
        
    
    #Pegamos la mascara del objeto recortado, en la mascara de la nueva imagen en la posicion que se ha indicado
    mask_new[y1:y2+1,x1:x2+1]=mask_obj[:,:] #el +1 es porque en python el indice final indicado no se cuenta
            
    #Pegamos tambien la imagen foreground del objeto recortado en la imagen foreground nueva, para tener una imagen con todo 0s excepto
    #los pixeles que pertenecen al objeto:
    foreground_new[y1:y2+1,x1:x2+1]=foreground_obj[:,:]
#    plt.imshow(foreground_new)
   
    #En caso de querer hacer blend de la imagen del objeto con el fondo, se aplica la funcion creada:
    if(blend_obj==True):
        
        outImage=blending(image_BG.astype('uint8'), foreground_new, mask_new, num_capas=3, grosor_capa=1)
    
    #Sino se pega la imagen del objeto enmascarado sobre el fondo directamente:    
    else:
        #Multiplicamos el fondo con la nueva mascara INVERTIDA para que los pixeles del objeto pegado se queden a 0:
        alpha=np.stack([mask_new,mask_new,mask_new],axis=2) #Como la mascara solo es un canal hay que convertirla a tres canales (RGB)
        background = cv2.multiply((1 - alpha).astype('uint8'), image_BG.astype('uint8'))
        
        #Sumamos la imagen foreground y background para obtener la nueva imagen con el objeto pegado
        outImage = cv2.add(foreground_new, background)
        
    
    #Mostramos el resultado en caso de que se requiera:
    if(muestra_resultado==True):
        plt.subplot(121) #crea un grid de 1fila,2columnas,seleecionando el elemento 1 de ese grid
        outImage_RGB=cv2.cvtColor(outImage, cv2.COLOR_BGR2RGB)
        plt.imshow(outImage_RGB)
        plt.subplot(122) #crea un grid de 1fila,2columnas,seleecionando el elemento 2 de ese grid
        plt.imshow(mask_new, 'gray')
        plt.show()
        
    return outImage, mask_new
    


def mouseCallBack(event, x, y, flags, param):
    """
    Funcion callback del raton que se ejecutara siempre que ocurra un evento en el raton
    en cv2 hay enumerados una serie de eventos que podemos tratar para averiguar que evento ha ocurrido
    Los argumentos de esta funcion deben ir en ese orden para que se almacenen ahi las lecturas del raton.
    
    Con esta funcion modificamos el valor de variables globales por lo que no devuleve ningun valor.
    """
    
    #Definimos como globales las variables que queremos que se modifiquen fuera de esta funcion o variables externas 
    #que queremos leer desde dentro de esta funcion
    global scale, image_BG
    global image_obj_orig, mask_obj_orig
    global image_obj_rescaled, mask_obj_rescaled
    global image_new, mask_new
    global resultado, resultado_ant
    global dir_img_FG
    
    #En caso de haber presionado el boton izquierdo del raton, se guardara como imagen
    #de fondo la imagen calculada con el objeto pegado, y se guarda la mascara nueva junto con las anteriores
    if event == cv2.EVENT_LBUTTONDOWN:
        
        #Pegamos el objeto en la imagen de fondo APLICANDO BLENDING:
        image_new, mask_new=aplica_mascara(image_BG, image_obj_rescaled, mask_obj_rescaled, [x, y], blend_obj=True)
        
        #Actualizamos la imagen de fondo a la nueva calculada
        image_BG=image_new.copy()
        
        #Actualizamos el resultado anterior:
        resultado_ant=copy.deepcopy(resultado) #si los asignas directamente la variable apuntara al mismo espacio en memoria por lo que modificar uno modificaria el otro. Luego .deepcopy es necesario ya que al ser un clase sino las funciones creadas dentro de ella en dos objetos distintos de esa clase tambien era la misma
        
        #Extraemos la etiqueta del objeto que se va a pegar, a partir de la ruta de la imagen del objeto
        label_obj=dir_img_FG.split(os.sep)[-2]
        
        #Almacenamos el resultado nuevo:
        resultado.nuevo_objeto(dir_img_FG, image_new, mask_new, label_FG=label_obj, pos_xy=[x, y], scale=scale)
     
    #En caso de haber presionado el boton derecho del raton, se vuelve un paso atras eliminando el 
    #ultimo objeto pegado (si lo hubiese)
    if event == cv2.EVENT_RBUTTONDOWN: 
        
        resultado=copy.deepcopy(resultado_ant)
        
        #Actualizamos BG:
        #   (Si fuera la primera vez que se pone un objeto en el fondo se sigue haciendo bien puesto que habiamos incializado
        #   resultado_ant.image_masked con image_BG)
        image_BG=resultado_ant.image_masked.copy()
        image_new=image_BG.copy()
        
    #En caso de mover el raton se calcula la imagen y mascara con el objeto pegado, pero solo la representamos, 
    #no la guardamos(Tampoco se aplica blending para ahorrar tiempo ya que solo servira para visualizacion):
    if event == cv2.EVENT_MOUSEMOVE:

        image_new, mask_new=aplica_mascara(image_BG, image_obj_rescaled, mask_obj_rescaled, [x, y]) 
        
    #En caso de mover la ruleta del raton se amplia o disminuye la imagen del objeto que se desea pegar en la imagen
    #de fondo:
    if event == cv2.EVENT_MOUSEWHEEL:
        #Cuando mueves la ruleta del raton arriba o abajo el event es el mismo (10)
        #Pero en flags aparece un numero que es positivo en caso de hacerlo hacia
        #arriba y negativo en caso de hacerlo hacia abajo
        
        #Definimos el incremento que se va a realizar: 
        if(scale>0.5):   
            diff=0.02
        else:
            diff=0.01  
            
        #Si el giro de la ruleta ha sido hacia arriba    
        if(flags>0): 
            scale=scale+diff #Aumentas un 1% la imagen del objeto 
            
        #Si el giro de la ruleta ha sido hacia abajo:
        elif(flags<0 and scale>diff):
            scale=scale-diff #Reduces un 1% la imagen del objeto
            
            #si escala es demasiado pequenna:
            if(scale<diff):
                scale=diff
                
        #Reescalamos la imagen con la nueva escala respecto de la imagen original:
        image_obj_rescaled, mask_obj_rescaled=cambio_escala_obj(image_obj_orig, mask_obj_orig, scale=scale)
        
        #Calculamos la imagen con la nueva ecala del objeto para visualizarla inmediatamente(sin esperarar a mover el raton):
        image_new, mask_new=aplica_mascara(image_BG, image_obj_rescaled, mask_obj_rescaled, [x, y])
#        print('Escala: {:.2f}'.format(scale))
        
        
class imagen_resultado:       
    
    """
    Clase creada par almacenar los resultados obtenidos del pegado de objetos en una imagen de fondo.  
    
    En cada objeto de la clase se guarda: 
        - tanto la imagen resultante, las mascaras de cada objeto y sus etiquetas
        - como la informacion necesaria para poder reproducir el resultado obtenido sin tener que guardar en memoria las mascaras
    
    Dispone de las siguientes funciones:
        -nuevo_objeto(): con la que se introduce la informacion de un nuevo objeto en la imagen
        -actualiza_mascaras(): con la que se modifican las mascaras de objetos anteriormente guardados para eliminar
                               los pixeles que esten ocluidos por el nuevo objeto pegado
        -representa_resultados(): Con la que se muestran los resultados de los objetos pegados hasta el momento
                                  (se representa la imagen RGB y las mascaras de cada objeto)
    """ 
    
    def __init__(self, image_BG_path):
        
        #Inicializamos cada instancia de la clase con la imagen de fondo y su ruta
        self.image_BG_path = image_BG_path
        
        #Inicializamos tambien listas vacias para las variables donde se almacenaran las imagenes, mascaras, y etiquetas
        #de los objetos pegados en la imagen
        self.images_FG_path=[] #ruta de Imagenes FG de objetos introducidos
        self.masks_FG=[]  #mascaras introducidas
        self.labels_FG=[] #Etiquetas de cada mascara de introducida
        self.pos_xy=[] #Posicion xy donde se indico poner la mascara
        self.scale=[] #Escala a la que se indico guardar la imagen del objeto
        
        
        #Inicializamos la imagen RGB actual con los objetos pegados hasta el momento (ninguno)
        self.image_masked=cv2.imread(image_BG_path)
        
        #Guardamos el tamanno de la imagen creada:
        self.image_size=self.image_masked.shape[:2] #Tamanno de la imagen que se va editar [rows,cols]
        
        
    def nuevo_objeto(self, image_FG_path, image_masked, mask_FG, label_FG, pos_xy, scale):
        """
        Funcion con la que se introduce la informacion de un nuevo objeto en la imagen
        
        Argumentos de entrada: 
            image_FG_path: ruta de la imagen del objeto que se ha introducido
            image_masked: imagen RGB resultado tras haber pegado el objeto
            mask_FG: mascara binaria del nuevo objeto pegado en la imagen
            label_FG: etiqueta del objeto introducido
            pos_xy: posicion del pixel central en el que se ha pegado el objeto sobre la imagen
            scale: escalado que se ha hecho a la imagen del objeto para hacerlo mas grande/pequenno
        """
        #Annadimos ruta de la nueva imagen
        self.images_FG_path.append(image_FG_path)
        
        #Modificamos mascaras anteriores con nuevas oclusiones y annadimos la nueva mascara
        self.actualiza_mascaras(mask_FG)
        self.masks_FG.append(mask_FG)
        
        #Annadimos etiqueta de la mascara, posicion central donde se coloco, y la escala utilizada de la imagen del objeto:
        self.labels_FG.append(label_FG)
        self.pos_xy.append(pos_xy)
        self.scale.append(round(scale, 2))
        
        #Imagen RGB resultado que se ha obtenido al pegar el nuevo objeto
        self.image_masked=image_masked.copy()
        
   
    def actualiza_mascaras(self, mask_new):
        """ 
        Funcion con la que se modifican las mascaras de objetos anteriormente guardados para eliminar
        los pixeles que esten ocluidos por la mascara del nuevo objeto pegado
        
        Argumentos de entrada:
            mask_new: mascara del nuevo objeto pegado
        
        """
        for i, mask_i in enumerate(self.masks_FG):
            
            #Aplicamos operacion logica: mask_i and not mask_new
            #para asi obtener mask_i sin los pixeles que ha superpuesto mask_new
            not_mask_new=cv2.bitwise_not(mask_new)
            self.masks_FG[i]=cv2.bitwise_and(mask_i, not_mask_new)

    def representa_resultado(self, num_cols=2):
        """
        Funcion con la que se representa la imagen resultante y las mascaras de los objetos pegados
        
        Argumentos de entrada:
            num_cols: numero de columnas en las que quieres que se representen las mascaras de los objetos
        """
        #Representamos la imagen RGB obtenida
        plt.figure(figsize=(15,15))
        plt.subplot(1, num_cols, 1) #Para que lo represente del mismo tamnno que mascaras
        plt.imshow(cv2.cvtColor(self.image_masked, cv2.COLOR_BGR2RGB)) #openCV guarda y lee imagenes como BGR por lo que hay que realizar una conversion
        plt.show()
        
        #Representamos cada mascara por separado:
        plt.figure(figsize=(15,15))
        idx=1
        for mask_i in self.masks_FG:
            
            plt.subplot(1, num_cols, idx) #representamos por filas en lugar de hacer matriz
            plt.imshow(mask_i, 'gray')
            idx=idx+1
            
            #si ya ha terminado la fila, pasa a la siguiente:
            if(idx==num_cols+1):
                plt.figure(figsize=(15,15))
                idx=1


def selecciona_categoria_nueva(path_all_categories, category_id, desired_categories=None, select_in_list=False):
    """
    Funcion para cambiar la categoria del objeto que se quiere pegar.
    Se puede o seleccionar la siguiente categoria presente en la lista de carpetas segun el category_id indicado,
    o se puede seleccionar una categoria manualmente por el terminal introduciendo su ID.
    
    ----------------------
    Parametros de entrada:
    ----------------------
    path_all_categories: ruta de la carpeta en la que estan las carpetas de cada categoria   
    category_id: ID de la anterior categoria utilizada, de manera que se seleeciona la siguiente en caso de
                 que select_in_list sea False
    desired_categories: para indicar con una lista solo algunas de las categorias para generar las imagenes. 
                        Si se indica como None coge todas las categorias que hay en la ruta path_all_categories.
    select_in_list: booleano que en caso de ser True muestra por el terminal una lista de las distintas
                    categorias con sus ID
                    
    ----------------------
    Parametros de salida:
    ----------------------
    category_id_new: ID de la nueva categoria
    obj_category: nombre de la nueva categoria
        
    """
    list_categories=sorted(os.listdir(path_all_categories))
    
    #En caso de que no se quieran todas las clases de objetos, cogemos solo las clases deseadas:
    if(not (desired_categories is None) ):
        list_categories=[categ_i for categ_i in list_categories if categ_i in desired_categories]
        
    
    #En caso de querer seleccionar una categoria determinada:
    if(select_in_list==True):
        
        print(' ')
        print('########################################################')
        print('                       ATENCION!!                       ') 
        print('########################################################\n')
        print('Introduce por terminal el ID de la categoria a la que desea cambiar\nde entre las que aparecen en la lista: ')
        
        #mostramos una lista con todas las categorias encontradas y sus ID:
        print('ID     Cat')
        print('--     ---')
        
        for ID, categ in enumerate(list_categories):
            print('{} ---- {}'.format(ID, categ))
            
        #Esperamos a que introduzca el usuario un ID:
        inp_categ=int(input('Introduzca la ID de la categoria que quiere seleccionar:  '))
        
        #Si la ID indicada es una valida se cambia el valor de category_id
        #En caso contrario se indica que no es valida la ID y se deja como estaba
        if(inp_categ <= len(list_categories)-1): 
            category_id_new=inp_categ
        else:
            print('ID introducida no valida')
            category_id_new=category_id
      
    #En caso de querer pasar a la siguiente categoria:
    else:
        category_id_new=category_id+1
        
        #Si ya se ha superado el ultimo ID de categorias se vuelve a la primera
        if(category_id_new > len(list_categories)-1):
            category_id_new=0
            
    return category_id_new, list_categories[category_id_new]         
            
    
    
    
    
def selecciona_objeto_nuevo(path_obj_category):
    """
    Funcion para cambiar el objeto que se quiere pegar a uno aleatorio de la carpeta indicada en path_obj_category.
    
    ----------------------
    Parametros de entrada:
    ----------------------
    path_obj_category: ruta de la carpeta de categoria de objetos que se quieren pegar
    
    ----------------------
    Parametros de salida:
    ----------------------
    NINGUNO: ya que modifica las variables globales en las que se almacena la imagen y mascaras del objeto que se quiere pegar.
    
    """
    #Definicion de variables globales que se van a modificar
    global scale
    global image_obj_orig, mask_obj_orig
    global image_obj_rescaled, mask_obj_rescaled
    global dir_img_FG
    global image_BG
    
    #Leemos la lista de nombres de imagenes de objetos que hay en la ruta especificada: 
    file_names = sorted(os.listdir(path_obj_category))
    
  
    #Escogemos uno al azar de entre toda lista de nombres segun distrib uniforme:
    name_imgObj=np.random.choice(file_names)
    
    #Leemos la imagen del objeto:
    dir_img_FG=os.path.join(path_obj_category, name_imgObj)
    image_FG=cv2.imread(dir_img_FG)
    
    #Inicializamos la escala con el tamanno original de la imagen:
    scale=1.0 #Escala a la que esta la imagen del objeto (100% por defecto para que sea la imagen sin modificar)
    
    #Calculamos la imagen RGB del objeto con su mascara (la imagen RGB no es igual a la que le pasas ya que esta recortadda):
    image_obj_orig, mask_obj_orig=genera_mascara(image_FG)
    
    #Inicializamos la imagen reescalada con el valor de la original (caso en que scale=1.0)
    image_obj_rescaled=image_obj_orig.copy() #(.copy es porque si igualas dos variables a secas en python, el valor en memoria de las dos es el mismo solo que con distintos nombres, por lo que si modificas uno modificarias el otro)
    mask_obj_rescaled=mask_obj_orig.copy()

    #En caso de que la imagen del objeto sea mayor que la del fondo, se ajusta su tamnno al del fondo
    #Sino, la funcion devolvera la imagen original:
    image_obj_rescaled, mask_obj_rescaled, scale=ajusta_obj_a_fondo(image_BG.shape[:2], image_obj_orig, mask_obj_orig, image_obj_orig.shape[:2], scale)
    
    
def selecciona_fondo_nuevo(path_BGs, num_BG):
    """
    Funcion con la que se cambia el fondo a otro nuevo para empezar a crear otra imagen nueva
    
    Modifica las variables globales en las que se almacena la imagen y mascaras de la imagen que se crea.
    
    ----------------------
    Parametros de entrada:
    ----------------------
    path_BGs: ruta de la carpeta donde guardamos los fondos
    num_BG: numero ID del fondo que quieres seleccionar 
    
    ----------------------
    Parametros de salida:
    ----------------------
    NINGUNO: ya que modifica las variables globales en las que se almacena la imagen y mascaras del objeto que se quiere pegar.
    
    
    """
    #Definimos las variables globales que van a ser leidas/escritas:
    global image_BG
    global image_new, mask_new
    global dir_img_BG
    
    #Obtenemos la lista de todos los fondos:
    list_BGs=sorted(os.listdir(path_BGs))
    
    #Como al ser strings los nombres y no enteros cuando llega a 100 o pone detras del 10 por lo que
    #lo ordenamos apropiadamente para que siga el orden correcto de los fondos 
    #(simepre que los fondos tengan un numero como nombre lo hara, sino no hara nada)
    try:
        list_BGs_int=np.array([int(BG_i.split('.')[0]) for BG_i in list_BGs])
        idx_reorder=list_BGs_int.argsort()
        list_BGs=list(np.asarray(list_BGs)[idx_reorder])
        list_BGs=[str(BG_i) for BG_i in list_BGs]
    except Exception:
        pass  
    
    #Actualizamos num_BG (corresponde con el ID de cada fondo). En caso de que haya superado el numero de fondos 
    #se vuelve al primer fondo
    num_BG=num_BG+1
    if(num_BG>len(list_BGs) or num_BG<1):
        num_BG=1
        
    #Leemos la imagen de fondo:
    dir_img_BG=os.path.join(path_BGs, list_BGs[num_BG-1])
    
    image_BG=cv2.imread(dir_img_BG)
    
    #Comprobamos si la imagen de fondo es mas grande que la resolucion del monitor, en ese caso la reescalamos a la resolucion del monitor
    #y la guardamos asi en memoria CON OTRO NOMBRE.
    modificada=False
    screensize=[1080, 1920] #NO MODIFICAR ESTOS VALORES YA QUE LAS IMAGENES DE FONDO ENTONCES CAMBIARIAN E IMAGENES ETIQUETADAS HASTA EL MOMENTO NO VALDRIAN
    while(screensize[0]<image_BG.shape[0] or screensize[1]<image_BG.shape[1] ): #Hacemos un bucle ya que dependiendo de la forma de la imagen se saldra por un
                                                                                #lado aunque ya lo hayamos reescalado para que otro lado si que este dentro
                       
        #Miramos que lado de la imagen de fondo es la que se sale:
        if(screensize[0]<image_BG.shape[0]):
            idx_sideOut=0
        elif(screensize[1]<image_BG.shape[1]):
            idx_sideOut=1
        
        #Reescalamos la imagen de fondo para que el lado mas grande de la imagen de fondo quepa en 
        #el monitor, PERO MANTENIENDO EL ASPECT RATIO:
        if(idx_sideOut==0): #imagen de fondo se sale de alto
            new_height=screensize[0]
            aspect_ratio = new_height / image_BG.shape[0]
            new_width=image_BG.shape[1]*aspect_ratio
            
            
        elif(idx_sideOut==1): #imagen de fondo se sale de ancho
            new_width=screensize[1]
            aspect_ratio = new_width / image_BG.shape[1]
            new_height=image_BG.shape[0]*aspect_ratio
        
        image_BG=cv2.resize(image_BG, (int(new_width),int(new_height)), interpolation=cv2.INTER_LANCZOS4)
        modificada=True
        
    if(modificada==True):
        #Guardamos la imagen con esta resolucion para no tener que volverla a reescalar en caso de volver a usar ese fondo:
        cv2.imwrite(dir_img_BG, image_BG)
        print('Modificada la resolucion del fondo "{}" a la del monitor'.format(dir_img_BG))
        
    #Inicializamos la imagen actual con el valor de la imagen de fondo para que se comience visualizando solo el fondo.
    #Tambien se inicializa las mascaras de esa imagen a None ya que todavia no habra ninguna
    image_new=image_BG.copy()
    mask_new=None #None ya que todavia no hay ningun objeto pegado
    
    return num_BG

def save_obj_pkl(path_name , obj_to_save):
    """
    Funcion para guardar cualquier objeto python en un fichero .pkl
    
    path_name: ruta en la que se quiere guardar el archivo
    obj_to_save: objeto python que se quiere guardar con el nombre especificado
    """
    with open(path_name, 'wb+') as f:
        pickle.dump(obj_to_save, f, pickle.HIGHEST_PROTOCOL)

def load_obj_pkl(path_name):
    """
    Funcion para cargar un fichero .pkl en el que se había guardado un objeto de python
    
    path_name: ruta del fichero .pkl que se quiere cargar
    """
    with open(path_name, 'rb') as f:
        return pickle.load(f)
    

    
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
        
    
def guarda_anotaciones_enPKL(images_created, images_created_path):
    """
    Funcion que guarda las mascaras y otra informacion de las imagenes creadas en un fichero .pkl
    Para ello se queda con solo algunos de los atributos guardados en los objetos de la 
    clase imagen_resultado, y los guarda en un diccionario
    
    """
    
    annotations_path=images_created_path+'/annotations.pkl'
    
    #Primero se comprueba si ya existe un archivo de anotaciones de otras imagenes anterior, y en ese caso se lee para partir de ese
    if(os.path.exists(annotations_path)):
        annotations = load_obj_pkl(annotations_path)
    else:
        annotations=dict()
        
    #Añadimos al diccionario de anotaciones todas la nuevas imagenes que han sido etiquetadas/anotadas:
    for key_image, value_image in images_created.items():
        
        #Primero comprobamos si ya existia una imagen con ese nombre en el fichero .pkl que guarda las anotaciones
        #y en caso afirmativo se pregunta si se desea sobreescribir
        if( (key_image in annotations.keys()) == True ):
            
            print('########################################################')
            print('                       ATENCION!!                       ') 
            print('########################################################')
                          
            inp=input('Ya existe una anotacion para una imagen con nombre {}\n ¿Desea sobreescribirla? [s/n]:  '.format(key_image) )
            
            if( inp == 'n'): #Si se indica que no, se pasa a la siguiente iteracion del for (siguiente imagen)
                continue
        
        #Introducimos la anotacion de la nueva imagen que guardaremos en el fichero .pkl:
        annotations[key_image]= {'image_BG_path': value_image.image_BG_path,
                                 'images_FG_path': value_image.images_FG_path,
                                 'image_size': value_image.image_size,
                                 'masks': value_image.masks_FG,
                                 'labels': value_image.labels_FG,
                                 'pos_xy': value_image.pos_xy,
                                 'scale': value_image.scale
                                }
        
    #Se guardan las anotaciones en un fichero .pkl
    save_obj_pkl(annotations_path, annotations)
 

def guarda_anotaciones_enJSON(images_created, images_created_path):
    """
    Funcion que guarda las mascaras y otra informacion de las imagenes creadas en un fichero .pkl
    Para ello se queda con solo algunos de los atributos guardados en los objetos de la 
    clase imagen_resultado, y los guarda en un diccionario
    
    """
    
    annotations_path=images_created_path+'/annotations.json'
    
    #Primero se comprueba si ya existe un archivo de anotaciones anterior de otras imagenes creadas, 
    #y en ese caso se lee para partir de ese
    if(os.path.exists(annotations_path)):
        
        annotations = load_obj_JSON(annotations_path)
        
    else:
        annotations=dict()
        
    #Añadimos al diccionario de anotaciones todas la nuevas imagenes que han sido etiquetadas/anotadas:
    for key_image, value_image in images_created.items():
        
        #Primero comprobamos si ya existia una imagen con ese nombre en el fichero .pkl que guarda las anotaciones
        #y en caso afirmativo se pregunta si se desea sobreescribir
        if( (key_image in annotations.keys()) == True ):
            
            print('########################################################')
            print('                       ATENCION!!                       ') 
            print('########################################################')
                          
            inp=input('Ya existe una anotacion para una imagen con nombre {}\n ¿Desea sobreescribirla? [s/n]:  '.format(key_image) )
            
            if( inp == 'n'): #Si se indica que no, se pasa a la siguiente iteracion del for (siguiente imagen)
                continue
                      
        
        #Introducimos la anotacion de la nueva imagen que guardaremos en el fichero .pkl:
        annotations[key_image]= {'image_BG_path': value_image.image_BG_path,
                                 'images_FG_path': value_image.images_FG_path,
                                 'image_size': value_image.image_size,
                                 'labels': value_image.labels_FG,
                                 'pos_xy': value_image.pos_xy,
                                 'scale': value_image.scale
                                }
     
    
    #Se guardan las anotaciones en un fichero .pkl
    save_obj_JSON(annotations_path, annotations)


###########################################################################
#    EJECUCION DEL PROGRAMA PRINCIPAL EN CASO DE EJECUTAR EL SCRIPT
###########################################################################
    
if __name__ == "__main__":
    print(main.__doc__)
    main()
    