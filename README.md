### TODO week 2.5-2.11
- [] train on gpu model 1
- [] add argument parser for trainer and evaluator 
- [] test speed for predictor, most likely needs to speed up 
- [] read Inception Resnet paper
- [] read SMAC and Spearmint 
- [] might implement SMAC or Spearmint


### Dependencies
(after installing miniconda which comes with essential python packages)
- Tensorflow (added path to CUDA toolkit)
- cv2
- dlib (for face detection and alignment)


### Dataset:
#### UTF faces (aligned and cropped):
https://susanqq.github.io/UTKFace/

The labels of each face image is embedded in the file name, formated like [age]_[gender]_[race]_[date&time].jpg

- [age] is an integer from 0 to 116, indicating the age
- [gender] is either 0 (male) or 1 (female)
- [race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
- [date&time] is in the format of yyyymmddHHMMSSFFF, showing the date and time an image was collected to UTKFace

gender_labels: 0: male, 1: female
{'0': 12391, '1': 11317})
race_labels:
Counter({'0': 10078, '1': 4528, '2': 3434, '3': 3976, '4': 1692})

augmentation:
0: no, 1: x2, 2: x3, 3: x3, 4: x4
23708 - 38505
Counter({'0': 10078, '1': 9056, '2': 13736, '3': 15904, '4': 8460})

38505 - train


### Main scripts
make_features.py
- create tfrecord files for training, validation and test data
- data augmentation to balance the representation each race in the total population


multitask_model.py
- data: read, decode data from .tfrecord; and preprocess the data for Inception Resnet
- build model: load network graph and add layers
- a few options of how to add layers, `add_layer` as the most basic one


trainer.py
- set training parameters and log directory
- train the model

evaluator.py
- evaluate trained model

predictor.py
Important dependencies:
1. dlib.
Install on Windows with instruction from https://github.com/charlielito/install-dlib-python-windows.
2. cv2
Also, might have to use an older version of python.
- preprocess: chop and align the face
- predict use saved model

---- model 1 - after 400 steps of batch_size = 8
INFO:tensorflow:Average gender Accuracy: 0.49626866
INFO:tensorflow:Average race Accuracy: 0.19216418
INFO:tensorflow:Average loss: 14.95729
