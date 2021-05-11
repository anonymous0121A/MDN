# MDN
Code for MDN
#### Datasets
Please unzip the rar files in ``Datasets\datasetName\mats`` to get large datasets (ML-10M, Netflix, Foursquare) first .
#### Model Training
Run rating prediction by
```
python labcode.py [save_model_name] [load_model_name]
```
Run location-based recommendation by
```
python labcode_locationPred.py [save_model_name] [load_model_name]
```
The dataset to train on and model parameters should be specified in ``Params.py``, which also contains the tuned hyper-parameters for each datasets. Please update the hyper-parameter list before you run another dataset. Default parameters are for ML-1M.
#### Trained Models
For each dataset, there are one trained model in ``Models\`` and a training log file in ``History\``.
To view the training log, specified the model name in ``ResAnalyzer.py`` and run
```
python ResAnalyzer.py
```
