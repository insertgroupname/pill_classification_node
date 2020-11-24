# pill_classification_node
  
  This project is using expressJS with PythonShell node module in order to run python file.
  
## Installation
```
$ npm i
```
## Start
```
$ npm start
```
### Color classification
If you want to add more class for traning color data,
	you can delete the file `training.data` in 
	```
	pill_classification_node/server/pyfol/shape_predict/func/src/color_recognition_api/
	```<br>
The `traning.data` will be generated after you run for another time.

### Document so far

Most preprocessing, shape prediction will be in this file<br>
This file will run magic code that will predict the shape

```
pill_classification_node/server/pyfol/shape_predict/func/fed.py
```
<p>Then, the picture will be throw in to this file and the color prediction will start from here</p>

```
pill_classification_node/server/pyfol/shape_predict/func/src/colorPredictor.py
```
<p>It will also run two file</p>

`pill_classification_node/server/pyfol/shape_predict/func/src/color_recognition_api/knn_classifier.py`<br>

and <br>

`pill_classification_node/server/pyfol/shape_predict/func/src/color_recognition_api/color_histogram_feature_extraction.py`<br>

### Stronk members
Mr.KIATISAK PETHOR [EEarth1270](https://github.com/EEarth1270)<br>
Mr.CHAICHET PHAIBUNWITTHAYASAK [mrforgotten](https://github.com/mrforgotten)<br>
Mr.VORAPOL CHAROENKITMONGKOL [flukerbooker](https://github.com/flukerbooker)<br>
	
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
