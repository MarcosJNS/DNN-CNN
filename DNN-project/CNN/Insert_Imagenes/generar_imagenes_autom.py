# -*- coding: utf-8 -*-
"""
Otra version de "generar_imagenes.py" que en vez de tener que crear manualmente la imagen
de manera totalmente automatica colocando los objetos aleatoriamente. Tiene el problema
de que al no guardar realismo las imagenes, la red neuronal obtiene resultados mucho peores 
que si las creas de manera realista.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

############################################################################################
#                               CODIGO DEL PROGRAMA PRINCIPAL
############################################################################################

def main():
    """
    PROGRAMA PRINCIPAL
    
    Al ejecutarlo se crean el numero de imagenes sinteticas indicado en num_images, 
    con un numero de objetos por imagen aletarorio entre 5 y 10 (indicado en num_objects), 
    y con una superposición entre objetos menor que la indicada con threshold_superposition

    """
    
    #Definimos e incializamos variables globales que queremos que sean leidas/escritas desde cualquier parte del codigo
    global image_BG
    global scale
    global image_obj_orig, mask_obj_orig
    global image_obj_rescaled, mask_obj_rescaled
    global dir_img_FG, dir_img_BG
    global image_new, mask_new    
    global resultado, resultado_ant
    
    #Indicamos la ruta donde queremos almacenar las imagenes creadas: 
    images_created_path='train_auto' 
    if(not os.path.exists(images_created_path)):
        os.mkdir(images_created_path)
    
    #Creamos un diccionario donde almacenaremos el resultado total con tadas las imagenes creadas:
    images_created=dict()

    #Indicamos la ruta donde estan almacenados los fondos:
    path_BGs='BG'
    
    #Indicamos la ruta donde estan almacenados los objetos:
    path_all_categories='objects'
        
    #Definimos el numero de imagenes sinteticas que se quiere crear:
    num_images=1000
    cnt_images=0 #contador de imagenes creadas
    
    #Bucle principal en el que se van creando las imagene sinteticas:
    while(cnt_images<num_images):

        #Caragamos la imagen de fondo :
        selecciona_fondo_aleatorio(path_BGs)
        
        #Creamos una instancia de la clase creada imagen_resultado, para almacenar los resultados obtenidos al pegar objetos
        resultado=imagen_resultado(dir_img_BG)

        #Seleccionamos el numero de objetos por imagen aleatoriamente:
        num_objects=np.random.choice(np.arange(5, 10+1))
        cnt_objects=0 #contador de objetos pegados
        
        while(cnt_objects<num_objects):
            #Cargamos una categoria de objeto:
            obj_catgory=selecciona_categoria_aleatoria(path_all_categories, exclude=['fork', 'egg', 'orange', 'potato'])
            path_obj_category=os.path.join(path_all_categories, obj_catgory)
            
            #Cargamos un objeto de la categoria seleccionada:
            images_to_exlude=['objects\\potato\\25.png',
                              'objects\\potato\\27.png',
                              'objects\\potato\\28.png',
                              'objects\\potato\\34.jpg',
                              'objects\\pan\\32.jpg',
                              'objects\\pan\\41.jpg']
            
            selecciona_objeto_nuevo(path_obj_category, exclude=images_to_exlude) #No devuelve nada ya que modifica variables globales
            
            #Seleccionamos aleatoriamente una escala del objeto:
            scale_ini=scale #Guardamos la escala inicial a la que se ha cargado el objeto (sera 1 o menor)
            scale_reduction=np.random.choice(np.arange(1.5, 2.5+0.1, 0.1))
            scale=round(scale_ini/scale_reduction, 2)
            
            #Reescalamos la imagen con la nueva escala respecto de la imagen original:
            image_obj_rescaled, mask_obj_rescaled=cambio_escala_obj(image_obj_orig, mask_obj_orig, scale=scale)
            
            #Seleccionamos aleatoriamente una posición en la que pegar el objeto. Se deja un marco en el que no se
            #puede colocar muyu cerca del borde para intentar no ocluir demasiado las imagenes:
            offset_x=int(image_BG.shape[1]*0.05)
            offset_y=int(image_BG.shape[0]*0.05)
            x=int(np.random.choice(np.arange(offset_x, image_BG.shape[1]-offset_x)))
            y=int(np.random.choice(np.arange(offset_y, image_BG.shape[0]-offset_y)))
            
            #Pegamos el objeto en la imagen de fondo:
            image_new, mask_new=aplica_mascara(image_BG, image_obj_rescaled, mask_obj_rescaled, [x, y])
            
            #Comprobamos si la localizacion del nuevo objeto calculado no ocluye en exceso a la máscaras anteriormente pegadas
            excess_superposition=comprueba_superposicion(mask_new, resultado, threshold_superposition=0.05) #solo se pegara si no ocluye mas de un 5% de las mascaras anteriores
            
            #Si no ocluye en exceso a ninguna mascara se añade definitivamente el nuevo objeto:
            if(excess_superposition == False):
                
                #Actualizamos la imagen de fondo a la nueva calculada
                image_BG=image_new.copy()
                
                #Extraemos la etiqueta del objeto que se va a pegar, a partir de la ruta de la imagen del objeto
                label_obj=dir_img_FG.split(os.sep)[-2]
            
                #Almacenamos el resultado nuevo:
                resultado.nuevo_objeto(dir_img_FG, image_new, mask_new, label_FG=label_obj, pos_xy=[x, y], scale=scale)
                
                #Sumamos uno al contador de objetos pegados:
                cnt_objects=cnt_objects+1
                
                #Representamos la imagen creada con sus mascaras:
                #resultado.representa_resultado(num_cols=3)

        #Aumentamos el contador de imagenes creadas
        cnt_images=cnt_images+1
        
        #Guardamos la imagen creada:
            #Cogemos el timestamp para annadirselo al nombre, y asi no sobreescribir otra imagen:
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%Hh%Mm%Ss")
        
            #Annadimos el timestamp al nombre delante para que las ultimas imagenes creadas se
            #queden al final de la carpeta. El nombre del fondo lo leemos para asegurarnos que 
            #coicide con el fondo que se ha puesto
        name_BG=(dir_img_BG.split(os.sep)[-1]).split('.')[0]
        img_name=timestamp+'_'+name_BG+'.png'
         
            #Almacenamos la imagen resultado en la variable images_created  y guardamos la imagen
            #en la carpeta especificada:                     
        images_created[img_name]=resultado
        cv2.imwrite(os.path.join(images_created_path,img_name), resultado.image_masked)
    

    #Tras finalizar la creación de todas las imagenes, guardamos las anotaciones de las imagenes 
    #creadas en un fichero .json 
    if(len(images_created) != 0):
        
        #Se guardan en un archivo .json todas las anotaciones de las imagenes realizadas
        #En el fichero json no se annaden las mascaras creadas, solo se guarda la informacion necesaria
        #para volver a generar la misma imagen guardada y sus mascaras.
        #Si ya habia en la carpeta indicada un archivo de anotaciones json, se añaden las anotaciones a las ya existentes 
        guarda_anotaciones_enJSON(images_created, images_created_path)
    
    #Cerramos las ventanas de opencv
    cv2.destroyAllWindows()
    

      
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
        
        
def aplica_mascara(image_BG, image_obj, mask_obj, pos_xy, muestra_resultado=False):
    
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
   
    #Multiplicamos el fondo con la nueva mascara INVERTIDA para que los pixeles del objeto pegado se queden a 0:
    alpha=np.stack([mask_new,mask_new,mask_new],axis=2) #Como la mascara solo es un canal hay que convertirla a tres canales (RGB)
    background = cv2.multiply((1 - alpha).astype('uint8'), image_BG.astype('uint8'))
#    plt.imshow(background)
    
    #Sumamos la imagen foreground y background para obtener la nueva imagen con el objeto pegado
    outImage = cv2.add(foreground_new, background)
    
    #Mostramos el resultado en caso de que se requiera:
    if(muestra_resultado==True):
        plt.subplot(121) #crea un grid de 1fila,2columnas,seleecionando el elemento 1 de ese grid
        plt.imshow(outImage)
        plt.subplot(122) #crea un grid de 1fila,2columnas,seleecionando el elemento 2 de ese grid
        plt.imshow(mask_new, 'gray')
        
        
    return outImage, mask_new
    
              
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

def comprueba_superposicion(mask_new, actual_image_result, threshold_superposition=0.05):
    """ 
    Funcion con la que se comprueba si la superposición entre la nueva máscara y las anteriores pegadas
    es mayor que un cierto umbral
    
    Argumentos de entrada:
        mask_new: mascara del nuevo objeto que se quiere pegar
        actual_image_result: objeto de la clase 'imagen_resultado' en el que estan almacenadas las mascaras pegadas hasta el momento
        threshold_superposition: valor umbral (0 a 1) que decide si un objeto esta muy superpuesto o no, en función de si el porcentaje
                                 ocluido de la máscara anterior es mayor que un cierto valor umbral en caso de pegar el nuevo objeto 
      
    Argumentos de salida:
        excess_superposition: booleano que indica si nueva mascara ocluye demasiado a alguna de las anteriores o no
        
    """
    excess_superposition=False
    
    for i, mask_i in enumerate(actual_image_result.masks_FG):
        
        #Calculamos el area de mask_i
        area_mask_i = cv2.countNonZero(mask_i)
        
        #Aplicamos operacion logica: mask_i and mask_new
        #para asi obtener los pixeles coincidentes en mask_i y mask_new
        mask_intersection=cv2.bitwise_and(mask_i, mask_new)
        
        #Calculamos el area de la interseccion entre mask_i y mask_new:
        area_mask_intersection = cv2.countNonZero(mask_intersection)
        
        #Comprobamos si el porcentaje ocluido de la máscara anterior 
        #es mayor que un cierto valor umbral en caso de pegar el nuevo objeto
        if((area_mask_intersection/area_mask_i)>threshold_superposition):
            excess_superposition=True
            break
    
    return excess_superposition        
      
      
def selecciona_categoria_aleatoria(path_all_categories, exclude=None):
    """
    Funcion para cambiar la categoria del objeto que se quiere pegar.
    
    ----------------------
    Parametros de entrada:
    ----------------------
    path_all_categories: ruta de la carpeta en la que estan las carpetas de cada categoria   
    exclude: lista con el nombre de las clases a excluir (example: exclude=['fork'])
    ----------------------
    Parametros de salida:
    ----------------------
    name_categ: nombre de la nueva categoria
        
    """
    list_categories=sorted(os.listdir(path_all_categories))
        
    #Quitamos las clases que se hayan indicado en exclude, si las hay
    if(exclude is not None):
        list_categories=[categ_i for categ_i in list_categories if categ_i not in exclude]
        
    #Se elige una de las categorias aleatoriamente:
    name_categ=np.random.choice(list_categories)
            
    return name_categ

  
  
    
def selecciona_objeto_nuevo(path_obj_category, exclude=None):
    """
    Funcion para cambiar el objeto que se quiere pegar a uno aleatorio de la carpeta indicada en path_obj_category.
    
    ----------------------
    Parametros de entrada:
    ----------------------
    path_obj_category: ruta de la carpeta de categoria de objetos que se quieren pegar
    exclude: lista con las rutas de las imagenes de objetos a excluir (example: exclude=['objects\\potato\\25.png'])
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
    
    #Quitamos las imagenes que se hayan indicado en exclude, si las hay
    if(exclude is not None):
        file_names=[file_i for file_i in file_names if os.path.join(path_obj_category, file_i) not in exclude]
    
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



def selecciona_fondo_aleatorio(path_BGs):
    """
    Funcion con la que se cambia el fondo a otro nuevo para empezar a crear otra imagen nueva
    
    Modifica las variables globales en las que se almacena la imagen y mascaras de la imagen que se crea.
    
    ----------------------
    Parametros de entrada:
    ----------------------
    path_BGs: ruta de la carpeta donde guardamos los fondos
    
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

    name_BG=np.random.choice(list_BGs)
    
    #Leemos la imagen de fondo:
    dir_img_BG=os.path.join(path_BGs, name_BG)
    
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
     
    
    #Se guardan las anotaciones en un fichero .json
    save_obj_JSON(annotations_path, annotations)


###########################################################################
#    EJECUCION DEL PROGRAMA PRINCIPAL EN CASO DE EJECUTAR EL SCRIPT
###########################################################################
    
if __name__ == "__main__":
    main()
    