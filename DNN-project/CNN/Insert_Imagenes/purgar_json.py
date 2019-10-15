# -*- coding: utf-8 -*-
"""
Script para eliminar las imagenes indicadas por su nombre en la lista imgs_to_delete

Elimina las imagenes de la carpeta y tambien sus etiquetas del script .JSON

"""
import os
import glob
import json

#Directorio en el se encuentran los archivos
ROOT_DIR = os.path.abspath("")
dataset_asistenteSteak=os.path.join(ROOT_DIR,"train_steak/*.png").replace("\\","/")




#Encuentra todos lo archivos (png) en este caso
flist = glob.glob(dataset_asistenteSteak)
print(flist)




data_sheet_path=os.path.join(ROOT_DIR,"train_steak/annotations.json").replace("\\","/")
new_data_sheet_path=os.path.join(ROOT_DIR,"train_steak/Nannotations.json").replace("\\","/")


#abrimos el json previo para copiar las características de las imágenes 
with open(data_sheet_path, 'rb') as f:
    anotations=json.load(f)

#Si ya se ha hecho una copia anterior la elimina para actualoizarla
if(os.path.exists(new_data_sheet_path)):
        os.remove(new_data_sheet_path)
else:
    Nannotations=dict()
    
#Copia en el nuevo archivo solo las imágenes existentes
for image_i in flist:
    image_i=os.path.basename(os.path.normpath(image_i))
    Nannotations[image_i]=anotations[image_i]

#Guardamos el nuevo archivo    
    with open(new_data_sheet_path, 'w+', encoding="utf8") as f:  
        json.dump(Nannotations, f, sort_keys=True, indent=4)
   