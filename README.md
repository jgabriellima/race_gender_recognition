## Dependencies
(after installing miniconda which comes with essential python packages)
1. Tensorflow (added path to CUDA toolkit)


## Dataset
### UTF faces (aligned and cropped):
https://susanqq.github.io/UTKFace/

The labels of each face image is embedded in the file name, formated like [age]\_[gender]\_[race]\_[date&time].jpg

- [age] is an integer from 0 to 116, indicating the age
- [gender] is either 0 (male) or 1 (female)
- [race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
- [date&time] is in the format of yyyymmddHHMMSSFFF, showing the date and time an image was collected to UTKFace

Handle error files from UTKFace:
``` 
mv 39_1_20170116174525125.jpg.chip.jpg 39_0_1_20170116174525125.jpg.chip
mv 61_1_20170109150557335.jpg.chip.jpg 61_1_3_20170109150557335.jpg.chip
mv 61_1_20170109142408075.jpg.chip.jpg 61_1_1_20170109142408075.jpg.chip
```            

### Raw
**gender_labels**   
0: male, 1: female  
{'0': 12391, '1': 11317}

**race_labels**  
White, Black, Asian, Indian, and Others  
{'0': 10078, '1': 4528, '2': 3434, '3': 3976, '4': 1692}

### Augmentation
0: no, 1: x2, 2: x3, 3: x3, 4: x4  

- **train genders**  
Counter({'0': 22188, '1': 21128})    
- **val gender**  
Counter({'0': 2473, '1': 2336})  
- **train races**  
Counter({'3': 10809, '2': 9270, '0': 9057, '1': 8156, '4': 6024})  
- **val race**  
Counter({'3': 1116, '2': 1032, '0': 1021, '1': 896, '4': 744})  


### Scripts
```
make_features.py
```
Functionality: 
- get images from directory: `UTKFace/*.jpg`
- split data in to training, validation and test set
- data augmentation to balance the representation each race in the total population
- save to `*.tfrecords* files

```
multitask_model.py
```
Functionality: 
- read_and_decode: read, decode data from training `.tfrecord`; and preprocess the data ()
- build_model: load network graph and add layers
- a few options of how to add layers, `add_layer` as the most basic one, `add_layer_v2` added additional FC before auxiliary branches
- compute losses

```
trainer.py

python trainer.py --project_dir /data/gender_race_face/ --model_name model1_gender_0001 --learning_rate 0.0001 --num_epoch 2
```  
- set training parameters and log directory
- train steps
- log to tensorboard 

```
evaluator.py

python evaluator.py --project_dir /data/gender_race_face/ --model_name model1_gender_0001 
```
Functionality:  
- evaluate trained model
- log to tensorboard

```
predictor.py
```
Important dependencies:
1. dlib.
Install on Windows with instruction from https://github.com/charlielito/install-dlib-python-windows.
2. cv2
Also, might have to use an older version of python.
- preprocess: chop and align the face
- predict use saved model

## Baseline models 
### model1_gender 
baseline: eval acc after 2 epoches 0.001 lr   
INFO:tensorflow:Average gender Accuracy: 0.601368   

0.0001 lr   
INFO:tensorflow:Average gender Accuracy: 0.932214   

### model1_race
0.0001 lr   
INFO:tensorflow:Average race Accuracy: 0.864428   

## Experiments 
#### model 1 - after 400 steps of batch_size = 8
INFO:tensorflow:Average gender Accuracy: 0.49626866   
INFO:tensorflow:Average race Accuracy: 0.19216418   
INFO:tensorflow:Average loss: 14.95729   

#### model 1 revised 
add 128 FC layer before prediction layer 

#### model 2 
change to xavier initializer

#### model 3
add batch normalization 


#### model 1 0.00001 epoch 2
- too slow to learn
INFO:tensorflow:Average gender Accuracy: 0.798507
INFO:tensorflow:Average race Accuracy: 0.653607
INFO:tensorflow:Average loss: 1.7886
 
#### model 0.0001 2 learning rate decay every 2000steps  
INFO:tensorflow:Average gender Accuracy: 0.829602   
INFO:tensorflow:Average race Accuracy: 0.770522   
INFO:tensorflow:Average loss: 1.17037   

#### model 3 0.0001 2000 steps batch_size 16

#### model 4 0.0001 added layer with dropout keep rate 0.8
- variance increase significantly  
- diminishing gradient? 

#### model 5 batch_norm lr 0.0001 
- step could be smaller   
- very good learning before steps 1500, but learning rate seems to be too large after    

#### model 6  augmented 0.001 3epoches

#### model 7 xavier initializer


#### model8_add addtional layer before auxilary networks and disregard bottleneck layer
INFO:tensorflow:Average gender Accuracy: 0.906095   
INFO:tensorflow:Average race Accuracy: 0.840796   

#### model9 set previous layers's learning rate 1/100 of current learning rate 
too much too learn 

#### model10 train seperate auxilinary networks with different lr 
- after 1 epochs: not much learning    
- mod: decay_steps - 2000 for race, 2000/2 for gender, combined opts, init/100, decay=1000   
- mod: learning_rate decay_step 1500 decay rate 0.9   

#### model11_add
add layer before auxilary layer, decay step 2000   
128 with 0.5 dropout    
overfitting    
train ~0.95/0.9 test ~0.75/0.55   

#### model11_add64   
32 neurons with 0.8 dropout    
a lot of fluctuation, cost keep dropping    
- might need to adjust learning rate to reduce (smoothed version seems to be linear) fluctuation 
- might have overfitted, check evaluation set 
- 16 neurouns, still overfitting after 4 epoches

16 units 0.5 dropout

init learning rate 0.01: worse performance at the beginning as compared to 0.001 lr    

#### model11_diff
different lr decay steps for gender and race    
gender - decay steps/4   

#### model12 
relu activation for added layer   

#### combined:
same learing rate roughout 

#### combined2
lower learning rate previous layer

#### combined3
lower learning rate prev layer, mid lr gender layer 

#### combined4
combined3 with batch normalization for fc layers
