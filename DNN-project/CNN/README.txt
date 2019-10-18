The folder 'CNN' is the module which uses the Mask-RCNN from Matterpport (Mask-RCNN) https://github.com/matterport/Mask_RCNN 
in order to detect and categorize objects.


----------------------------------
	     FOLDERS
----------------------------------

- Trained_Models:  In this folder it can be found a trained file which can be used fo detection if you want to give it a try.

- CNN_Dataset: Here there are different dataset used and saved with real images from RGB or RGBD cameras. The image tagging has been done with VIA tool, 
					not all the images been tagged already.

- Hand_Dataset: It is a datased with tagged hands with segmentation mask in the same format as the ones found in 'CNN_Dataset'. 

- Insert_Images: It contains some scripts to create synthetic image to speed up the process of dataset building.



- mAP-master: Library obtained from the repositoru (https://github.com/Cartucho/mAP) that can be used to calculate mAP metrics, introduced in PASCAL VOC 2011. 
					It is used to obtain AP metric for each class. 




----------------------------------
	   SCRIPTS
----------------------------------

- CNN_Lib.py: This script enables the use of the other ones. It is necessary for the convolutional neural network configuration, as well as loading datasets and 
					transform tag format to one admitted by  Mask-RCNN (Matterport). 
					There are some function implemented for results visualization too.

- CNN-Detection: File which has the function to run the Msk RCNN network on one of the videos recorded with the RGBD intel RealSense D415 Camera ('stored at 'CNN_dataset/tracking_videos'). 
						 The video files are in .bag format with an RGB video and a depth video to categorize objects. The resultant video are stored inside the same folder
						 in a created directory called (detections). But even more important the tracking data is stored in 
						
   					


- train_asistente_MaskRCNN.py: Script to train the Mask-RCNN model, the model configuration predefined in CNN_Lib is used along with the pretrained model. 
						 IMPORTANT, to avoid conflict each time we start a trainning the scrit creates a folder inside 'logs' to store 
						 the data and weights, we have to copy the pretrainned weights we wnt to use.
						 In this case I have upload one pretrained file in 'logs/log1' that will have to be copied to the file created by the 
						 script in 'asistente2019...' and then, run the program again.