# Encoding-Real-Inputs


## Tasks:
task -> keyword
- Classification -> classification
- Regression -> regression

## Datasets 
The datasets are fetched from OpenML and Scikit-Learn libraries
|Dataset|Keyword|Task|
|-------|-------|----|
|House | house_16h|Regression|
|Adult|adult|Classification|
|Gesture Phase Segmentation Processed|gesture|Classification|
|Microsoft|microsoft|Regression|
|Cover type|covtype|Classification|
|Otto Group Product|otto|Classification|
|Santander Customer Satisfaction|santander|Classification|
|Higgs|higgs|Classification|
|California Housing|california|Regression|


## Scripts: 
```console
python baseline.py --dataset {keyword} --task {regression|classification}
```