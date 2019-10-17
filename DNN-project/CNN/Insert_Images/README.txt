In this folder "Insert Images" there is a script called "generate_images" which allows you to create synthetic images.
This will dramatically shorten the time needed for the dataset creation.

En esta carpeta "asistente/InsertarImagenes" se ha desarrollado un programa python "generar_imagenes.py"
el cual te permite crear imagenes sinteticas ya etiqeutadas mediante el pegado de imagenes de objetos
sobre imagenes de fondo, de manera que las imagenes quedan ya etiquetadas con las mascara del objeto sobre el fondo.

For this purpose there are different folders containing the background ('BG'9 and several objects.

The images are not saved, instead we save the parameters from the picture to recreate the image again while the training take place. 
This way we can save space and training time.


---------------------------------------
	IMAGES FOLDER
---------------------------------------

-BG: Background selected images.


-BG_steak: Backgrounds used specifically for the action tracking purpose.

- objects: In this folder it can be found subdirectories with each of the classes we can introduce.The objects must be on top of a white background,
           that way the functions can segementate the object from the whole picture.
	   It is suggested to use"segmentaciones.py" or 
 	   "segmentacionesBlending.py"  in order to diminize the effect of bad segmentation (sometimes the automatic segmentation is far from perfect).

- train, train_auto, train_blend, train_steak: Folders with the created images (we can save the images itself if we want to).



----------------------------------
	   SCRIPTS
----------------------------------

- erase_images_autom.py: This script erases the designated images. It is important to use this code to do so because
				we have to eliminate both the image and the JSON tag info referred to the picture.

- generate_images: Main program to create new dataset, the instruction are inside the script.



- purgue_json.py: In case of mismatch between the JSON file and the dataset itself purgues the JSON.

- regenera_imgsConBlend.py: script que regenera un dataset que no habia sido creado aplicando blending al pegado de objetos
			   para así aplicarlo. 

- renombra_imagenes.py: script para renombrar las imagenes de la carpeta fondos por numeros. 
			IMPORTANTE!!!: no renombrar los fondos despues de haber creado ya imagenes con los nombres anteriores
 			               de fondos ya que sino esas imagenes ya no valdran.

- segmentaciones.py: script para visualizar como segmenta los objetos el programa antes de pegarlos sobre un fondo. La segmentacion la
		     realiza aplicando un umbral automatico para eliminar el fondo BLANCO. 
                     Para visualizar la segmentacion, pega el objeto segmentado sobre un fondo negro. En caso de que la segmentacion 
		     realizada deje huecos en el objetos (al tener partes blancas) es conveninete "pintar" esas partes con un editor de 
		     fotos como GIMP a un color mas grisaceo para que segmente el objeto correctamente.

- regenerate_imgs_With_Blend: funciona igual que "segmentaciones.py" pero aplicando blending entre el contrno del objeto y 
			     el fondo para que el fusionado de las dos imagenes no tenga tanto contraste.
