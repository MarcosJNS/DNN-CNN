----------------------------------
	   SCRIPTS
----------------------------------

- DNNAnalysis_Lib.py: This script holds the training and prediction for the DNN model (DNN model created with Tensorflow). 
          It also has some function to filter and standarize the data coming from the "CNN/CNN_Detectio" script. There is more 
          descriptive information inside the scripts, which I will keep adding to make the understanding process easier.
		

- main.py: The main program is really simple, it alludes to 'DNNAnalysis_Lib.py' to perform all the actions mentioned above. Depending on the object 
          we want to track it will create different folders.
