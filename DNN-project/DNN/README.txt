----------------------------------
	   SCRIPTS
----------------------------------

Información detallada de cada funciñon y/o script dentro del mismo.

- DNNAnalysis_Lib.py: Biblioteca que normaliza los datos y actualiza los modelos, entrenamiento del modelo MLP para los objetos que se estén realizando. 		

- main.py: Programa breve que manda órdenes a la biblioteca. Aquí hay diferentes combinaciones de órdenes para que hacer entrenamiento y predicción de datos.

-pipeline.py: Desarrolla y simplifica un poco las órdenes de 'main.py' antes de llamar a las funciones de la bilbioteca DNNAnalysis_Lib.py'.

-validation.py: Script que se usa para ver los métodos de evaluación de la red MLP (crea la matriz de confusión, 
Precision y recall, F1 Score...).

----------------------------------
	   FOLDERS
----------------------------------

-data_pan, data_hand: Se han implementado dos modelos de la red MLP para reconocer acciones de la sartén y de las manos. Hay dos carpetas para cada uno de ellos.

-Model_optimization: Esta carpeta contiene los pasos necesarios para realizar el proceso de K-fold_validation. Se usa para ver como de bien funciona el modelo dependiendo
del número de capas, datos y neuronas por capa que se quieran utilizar. Como es proceso largo, ya que entrena el mismo modelo muchas veces, se realizó ya un experimento para 
diferente número de capas y neuronas en el TFM. Los resultados están dentro de la carpeta con un notebook listo para ser ejecutado y visualizar los resultados.