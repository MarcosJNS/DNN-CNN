In this folder "Insert_Images" there is a script called "generate_images" which allows you to create synthetic images.
This will dramatically shorten the time needed for the dataset creation.

For this purpose there are different folders containing the background ('BG'9 and several objects.

The images are not saved, instead we save the parameters from the picture to recreate the image again while the training take place. 
This way we can save space and training time.


---------------------------------------
	IMAGES FOLDER
---------------------------------------

-BG: Background selected images.


-BG_steak: Backgrounds used specifically for the action tracking purpose.

- objects: In this folder it can be found subdirectories with each of the classes we can introduce.The objects must be on top of a white            background, that way the functions can segementate the object from the whole picture.
	   It is suggested to use"segmentations.py" or  "regenerate_imgs_with_blend.py"  in order to diminize the effect of bad    		   segmentation (sometimes the automatic segmentation is far from perfect).

- train, train_auto, train_blend, train_steak: Folders with the created images (we can save the images itself if we want to).



----------------------------------
	   SCRIPTS
----------------------------------

- erase_images_autom.py: This script erases the designated images. It is important to use this code to do so because
				we have to eliminate both the image and the JSON tag info referred to the picture.

- generate_images: Main program to create new dataset, the instruction are inside the script.



- purgue_json.py: In case of mismatch between the JSON file and the dataset itself purgues the JSON.

- regenera_imgsConBlend.py: This code generates a dataset applying blending between the sticked objects and the background on an image. 

- segmentations.py:  Script to visualize how the objects are segmented before sticking them to the background. As mentioned before the    				segmentation is done in a white background. in case the objects are not well segmented it would be 				      recommendable to use GIMP or other image editor to segmentate such picture

