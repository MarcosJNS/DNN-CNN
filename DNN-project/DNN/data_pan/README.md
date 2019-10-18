This is an example of action recognition for a pan with its subfolders.


----------------------------------
	   FOLDERS
----------------------------------

- classified: In this directory it is found the actions divided in 3 NaN( does not perform any action) place (is placing a pan) and remove (is removen a pan).
          The CSV creating with the tracking at 'CNN/CNN_detection.py' is saved directly here so it is important to rename the new files with action they perform. Each file for this 
           category must contain one action, no more.
		

- data_prediction: The prediction is made with csv with multiple actions stored in this folder, you can process more than one file at a time.
          The ones that are being currently use can be kept in 'data_prediction/waiting_for_pred'. With the function *pred_norm* inside 'main.py' we normalize the data for the prection and it creates a .csv file called
          'predict_'object'.csv'
          
          
- model_pan: Stores data from the DNN (FNN) training.


- raw_data: A folder to store data, it is created automatically to store not classified data that we want to keep from an object.

- train_ready: The directory has the training and validation datasets with two different purposes. 'Action_' means it is computed with the data from
            the folder 'classified' composed by files of single actions. 'Predict_' is made of tracking files with more than one action, which have to be separated manually to train the network, that is why this predict- .csv 
            files do not  have any previous filters.


-train_test_split: Normally the one action.csv files stored in 'classified' are processed in one function and also divided into val and train. The 'Predict_train' and 
'Predict_test' are not, so they need to be splitted with another function.  The results are kept in  'train_test_split/splitted', which you have to copy to 'train_ready if you wish to use them for training.       
            
