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
