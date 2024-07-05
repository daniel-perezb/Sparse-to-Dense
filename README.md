# Sparse to dense 

This is the opensource code for the following papers:
- Skirting Line Estimation Using Sparse to Dense Deformation: Banuelos, Daniel Perez, et al. 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2023

## Data used for training
Data of the difference fleeces used in training_files.json can be dowloaded from the following link:

https://github.com/daniel-perezb/IROS_webpage_fleece/tree/main/dataset

## Dependencies instalation
```
pip install numpy json5 scipy pynanoflann torch matplotlib scikit-learn
```

## Dense features
All dense features were extracted using 'dsift' from MATLAB. 
Implementation is explained in following link: https://www.vlfeat.org/overview/dsift.html

## Training 
1. Ensure that all the required files from 'training_files.json' are available.
2. Execute the script: 

```
python train_MLP.py
```

## Analysing new data
1. To analyse new data, dense and sparse features of the image pairs need to be extracted. The inliers after need to be saved as:
'sparse_features.json'
Outliers or dense correspondenes need to be saved as: 'dense_features.json' in current folder
2. Execute the script: 

```
python main.py
```

3. Filtered features are then saved as 'matrix_00.mat' in current folder and can be opened using matlab.

4. Threshold used for filter can be changed in 'main.py' in line, were .75 represents matches with above 75% confidence of inliers

```
indies = np.where(output[:, 1] > 0.75)
```
