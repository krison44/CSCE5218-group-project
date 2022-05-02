# Software Detection Prediction
The feasibility of the result of software development operations is assessed through a critical and costly review procedure in development programs. As project intricacies grow, individually finding software problems becomes a time-consuming and pricey effort. As a substitute to traditional abnormality identification, using automated classifiers to engage on problematic components lets the software developer dig further into the problem. Enhanced fault indicators may frequently locate a program that may be applied to a software consistency application in this scenario. As a result, there are several verified and created baseline classifiers. Simple predictors, particularly fault-detection capabilities, can be used with an ensemble technique to maximize productivity even further. The study's main goal is to see how useful base and ensemble prediction models are for fault diagnosis. The suggested research, which is being implemented in the PROMISE directory, emphasizes on identifying software problems employing Machine Learning as well as Deep Learning-based categorization approaches, then reviewing statistical results to choose the optimum matched model. The purpose of the research would be to show that ensemble classifiers may increase fault identification capability to a certain extent.


## Prerequisites
- Install the following packages
```
!pip install os
!pip install pandas
!pip install matplotlib
!pip install numpy
!pip install imblearn
!pip install tensorflow
!pip install keras
!pip install ensemble
!pip install seaborn
!pip install joblib
!pip install six
!pip install pydot
```
- Import the pollowing packages
```
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np
from imblearn.over_sampling import SMOTE

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Conv2D,Conv1D,Flatten,MaxPool2D

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

import seaborn as sb
import joblib

from sklearn.tree import export_graphviz
import six
import pydot
from sklearn import tree
``` 

## Datasets
```
ar_data: (428, 31)
mc_data: (9466, 40)
pc_data: (5589, 38)
cm_data: (498, 22)
kc_data: (2631, 22)
```

#### KC
- Pre-Processing
  - Before Pre-Processing 
    
    ![image](https://user-images.githubusercontent.com/4283187/166269839-02b07aaf-5ee2-4ca7-a683-d959f4bf6adc.png)
    ![image](https://user-images.githubusercontent.com/4283187/166270000-fa894c3b-e89b-42fb-a2fb-99c2287d6e5f.png)
    
  - After Pre-Processing
    
    ![image](https://user-images.githubusercontent.com/4283187/166269948-0bead166-f849-49b5-982c-a23bb7b3de46.png)
    ![image](https://user-images.githubusercontent.com/4283187/166270024-4dee791d-d9d1-45f8-a152-860c495bbd41.png)
  

- Base Model Test Result
  - Neural Network 
  
    ![image](https://user-images.githubusercontent.com/4283187/166270093-b9f2d24c-86a0-4430-9c93-19a666d3f618.png)
    ![image](https://user-images.githubusercontent.com/4283187/166270156-f72134bf-db4d-4d2f-9c41-57f5b692ecb1.png)
    ![image](https://user-images.githubusercontent.com/4283187/166270173-047543af-7a8d-4633-b47f-cff082312e34.png)
    ```
    Accuracy of Model: 0.7690355329949239
    Test Accuracy or Neural Network 0.7690355329949239
    ```
    
   - Random Forest
   
     ![image](https://user-images.githubusercontent.com/4283187/166279461-0f3515d6-c1b4-4567-951b-aa41073025dc.png)
     ![image](https://user-images.githubusercontent.com/4283187/166279658-93551df1-b24f-4d15-b275-ac4d532d5e9c.png)
     ```
      Accuracy of Model: 0.7766497461928934
      Test Accuracy of Random Forest 0.7766497461928934
     ```
     
   - Support Vector Machine

      ![image](https://user-images.githubusercontent.com/4283187/166283466-157edb35-5940-4beb-a6a7-13632b6926ef.png)
      ```
      Accuracy of Model: 0.8071065989847716
      Test Accuracy of SVC 0.8071065989847716
      ```
      
   - Convolutional Neural Network

      ![image](https://user-images.githubusercontent.com/4283187/166284680-9ac39bee-6a42-4bd8-8e9c-557193269fcd.png)
      ![image](https://user-images.githubusercontent.com/4283187/166284862-04ce5d0a-8ef8-4373-aae9-4739787323cf.png)
      ![image](https://user-images.githubusercontent.com/4283187/166284987-5afdec84-0297-4b1a-a3ce-c02bef0691ad.png)
      ```
      Accuracy of Model: 0.766497461928934
      Test Accuracy of CNN 0.766497461928934
      ```

- Ensemble Model Test Result
  
  ```
  KC Logistic Regression Accuracy: 0.7234848484848485
  KC Ada Boost Accuracy: 0.6515151515151515
  KC Bagging Accuracy: 0.7310606060606061
  ```
  
  
#### AR
- Pre-Processing
  - Before Pre-Processing 
    
    ![image](https://user-images.githubusercontent.com/4283187/166289511-3933c9af-144b-4da4-9487-0856905af763.png)
    ![image](https://user-images.githubusercontent.com/4283187/166289604-c48da978-233d-4801-9310-12f4ed822b38.png)

  - After Pre-Processing
    
    ![image](https://user-images.githubusercontent.com/4283187/166289572-cbf7e33c-488c-45aa-8369-ddde21ff1afd.png)
    ![image](https://user-images.githubusercontent.com/4283187/166289537-3713d614-056f-40a1-a718-06180cd1bde6.png)
  

- Base Model Test Result
  - Neural Network 
  
    ![image](https://user-images.githubusercontent.com/4283187/166289635-69f1fb98-7785-4909-b575-3328cac386cb.png)
    ![image](https://user-images.githubusercontent.com/4283187/166289642-72129b3d-d741-4f84-ab76-9b0108f676f4.png)
    ![image](https://user-images.githubusercontent.com/4283187/166289663-9bab1ed4-21fe-40b4-9531-8441f0c279ad.png)
    ```
    Accuracy of Model: 0.6818181818181818
    Test Accuracy or Neural Network 0.6818181818181818
    ```
    
   - Random Forest
   
     ![image](https://user-images.githubusercontent.com/4283187/166289709-50b7f337-0869-4f2e-8fa5-d74dcb7413a0.png)
     ![image](https://user-images.githubusercontent.com/4283187/166289724-dff26b99-1da1-424a-b012-a25408a6932c.png)
     ```
      Accuracy of Model: 0.8787878787878788
      Test Accuracy of Random Forest 0.8787878787878788
     ```
     
   - Support Vector Machine

      ![image](https://user-images.githubusercontent.com/4283187/166289790-95079907-b43a-46e5-9e84-462823c28c43.png)
      ```
      Accuracy of Model: 0.8636363636363636
      Test Accuracy of SVC 0.8636363636363636
      ```
      
   - Convolutional Neural Network

      ![image](https://user-images.githubusercontent.com/4283187/166289836-91c3c715-1608-4f91-8c9a-5e2c6868342f.png)
      ![image](https://user-images.githubusercontent.com/4283187/166289846-775d07d3-a416-462c-8fb2-2048883fbb12.png)
      ![image](https://user-images.githubusercontent.com/4283187/166289862-5552bffe-8b3c-4c7d-be72-57cf3fa82ca1.png)
      ```
      Accuracy of Model: 0.8636363636363636
      Test Accuracy of CNN 0.8636363636363636
      ```

- Ensemble Model Test Result
  
  ```
  AR Logistic Regression Accuracy: 0.7906976744186046
  AR Ada Boost Accuracy: 0.9069767441860465
  AR Bagging Accuracy: 0.8604651162790697
  ```
  
  
#### MC
- Pre-Processing
  - Before Pre-Processing
  - 
    ![image](https://user-images.githubusercontent.com/4283187/166289997-c4fadaf5-016a-4cbc-88a8-53b63014d17e.png)
    ![image](https://user-images.githubusercontent.com/4283187/166290110-1088e9c7-5b1c-4d16-8ae0-a820900e05fc.png)
    
  - After Pre-Processing
  
    ![image](https://user-images.githubusercontent.com/4283187/166290082-09b598a2-23ae-4216-8f13-446ae88120d9.png)
    ![image](https://user-images.githubusercontent.com/4283187/166290018-9f3af9bc-61f3-42b8-91bf-0639906cc1c0.png)
  

- Base Model Test Result
  - Neural Network 
  
    ![image](https://user-images.githubusercontent.com/4283187/166290188-1d950944-ccfb-4068-b478-72c74b67bc87.png)
    ![image](https://user-images.githubusercontent.com/4283187/166290203-ed5801d0-1e50-41e5-8ef3-a1f051a1af9a.png)
    ![image](https://user-images.githubusercontent.com/4283187/166290216-2036ca31-1580-4865-9d29-07b97c5b8555.png)
    ```
    Accuracy of Model: 0.9550827423167849
    Test Accuracy or Neural Network 0.9550827423167849
    ```
    
   - Random Forest
   
     ![image](https://user-images.githubusercontent.com/4283187/166290287-33778eea-2925-4210-82e9-1f2d3300c448.png)
     ![image](https://user-images.githubusercontent.com/4283187/166290308-e492f99e-2b26-4831-9827-5b179ead431e.png)
     ```
      Accuracy of Model: 0.9698581560283688
      Test Accuracy of Random Forest 0.9698581560283688
     ```
     
   - Support Vector Machine

      ![image](https://user-images.githubusercontent.com/4283187/166290385-577fbf6c-a12d-4e8e-b944-7bf3fc563ecc.png)
      ```
      Accuracy of Model: 0.9674940898345153
      Test Accuracy of SVC 0.9674940898345153
      ```
      
   - Convolutional Neural Network

      ![image](https://user-images.githubusercontent.com/4283187/166290451-0f5f51b2-c364-4cb2-b0a6-add8e32e45a7.png)
      ![image](https://user-images.githubusercontent.com/4283187/166290470-cf0f998a-c12b-4099-aaa5-00b16cf594ae.png)
      ![image](https://user-images.githubusercontent.com/4283187/166290487-7e7ef372-e9f3-48f3-8ef3-07c9807caee6.png)
      ```
      Accuracy of Model: 0.9911347517730497
      Test Accuracy of CNN 0.9911347517730497
      ```

- Ensemble Model Test Result
  
  ```
  MC Logistic Regression Accuracy: 0.9788806758183738
  MC Ada Boost Accuracy: 0.9355860612460402
  MC Bagging Accuracy: 0.9809926082365364
  ```
  
  
#### PC
- Pre-Processing
  - Before Pre-Processing 
    
    ![image](https://user-images.githubusercontent.com/4283187/166291138-6badfefb-3fae-493c-9e0d-946eceb32aa0.png)
    ![image](https://user-images.githubusercontent.com/4283187/166291207-2d5133fb-7c96-485b-8481-58eb64f087fb.png)
    
  - After Pre-Processing
    
    ![image](https://user-images.githubusercontent.com/4283187/166291249-a163e8f3-73f6-481d-af6c-9f8d8b1cc232.png)
    ![image](https://user-images.githubusercontent.com/4283187/166291300-11df682e-f29a-4b95-8924-c707be951e11.png)
  

- Base Model Test Result
  - Neural Network 
  
    ![image](https://user-images.githubusercontent.com/4283187/166291361-8e5d7a22-86d2-4c38-abc6-5824071b11f4.png)
    ![image](https://user-images.githubusercontent.com/4283187/166291377-d0eab4d3-6f6d-4325-9276-7fa33248c5cc.png)
    ![image](https://user-images.githubusercontent.com/4283187/166291400-b3512276-d4a5-4c03-b567-0209e9fef3cb.png)
    ```
    Accuracy of Model: 0.9690927218344965
    Test Accuracy or Neural Network 0.9690927218344965
    ```
    
   - Random Forest
   
     ![image](https://user-images.githubusercontent.com/4283187/166291467-5b814e11-239a-48fe-b941-bf9bbc638985.png)
     ![image](https://user-images.githubusercontent.com/4283187/166291484-058b7e78-c3b7-4084-a5e6-e617ab5bd7fe.png)
     ```
      Accuracy of Model: 0.9800598205383848
      Test Accuracy of Random Forest 0.9800598205383848
     ```
     
   - Support Vector Machine

      ![image](https://user-images.githubusercontent.com/4283187/166291526-a4a26f15-f68c-4e11-be3c-0be9ff35b52a.png)
      ```
      Accuracy of Model: 0.9581256231306082
      Test Accuracy of SVC 0.9581256231306082
      ```
      
   - Convolutional Neural Network

      ![image](https://user-images.githubusercontent.com/4283187/166291582-e1070d1c-5793-481f-8bda-cb4df62598e6.png)
      ![image](https://user-images.githubusercontent.com/4283187/166291607-e74e384c-b2b4-42a8-88a5-08d2ed73e77e.png)
      ![image](https://user-images.githubusercontent.com/4283187/166291621-d2e5f33e-8a9d-4f5a-8988-f232be0bfc4a.png)
      ```
      Accuracy of Model: 0.9920239282153539
      Test Accuracy of CNN 0.9920239282153539
      ```

- Ensemble Model Test Result
  
  ```
  PC Logistic Regression Accuracy: 0.9749552772808586
  PC Ada Boost Accuracy: 0.9713774597495528
  PC Bagging Accuracy: 0.9749552772808586
  ```
  
  
#### CM
- Pre-Processing
  - Before Pre-Processing 
    
    ![image](https://user-images.githubusercontent.com/4283187/166291736-b234f10d-d3e1-4f1e-9397-290df96df796.png)
    ![image](https://user-images.githubusercontent.com/4283187/166291761-15e13081-2ce5-49d1-b7d2-e947fcc47e0a.png)

  - After Pre-Processing
    ![image](https://user-images.githubusercontent.com/4283187/166291796-b384a003-6245-45da-b9f8-e9278180c2eb.png)
    ![image](https://user-images.githubusercontent.com/4283187/166291817-1fbda047-e8e3-4742-83a0-fc29da01f1e8.png)
  

- Base Model Test Result
  - Neural Network 
  
    ![image](https://user-images.githubusercontent.com/4283187/166291855-3d581862-c448-4665-a424-4449c04fc3dc.png)
    ![image](https://user-images.githubusercontent.com/4283187/166291872-be37f2e6-a088-4b37-89ee-e6b1e24a9a4b.png)
    ![image](https://user-images.githubusercontent.com/4283187/166291882-2fa7ab0d-1249-497a-9727-d1d61efa89fa.png)
    ```
    Accuracy of Model: 0.7875
    Test Accuracy or Neural Network 0.7875
    ```
    
   - Random Forest
   
     ![image](https://user-images.githubusercontent.com/4283187/166291929-3d5580ab-3cde-4dfe-80a8-b1403b78e67b.png)
     ![image](https://user-images.githubusercontent.com/4283187/166291946-dabd6e54-6425-4e12-9408-2b7bef242d55.png)
     ```
      Accuracy of Model: 0.7875
      Test Accuracy of Random Forest 0.7875
     ```
     
   - Support Vector Machine

      ![image](https://user-images.githubusercontent.com/4283187/166291980-590a8040-3b34-4045-b03c-9e38c2f3a877.png)
      ```
      Accuracy of Model: 0.8625
      Test Accuracy of SVC 0.8625
      ```
      
   - Convolutional Neural Network

      ![image](https://user-images.githubusercontent.com/4283187/166292040-dd5324ee-19f3-4ea2-a92d-0fd6aec354ec.png)
      ![image](https://user-images.githubusercontent.com/4283187/166292049-f8d13286-0001-4c59-af92-9d78ba5fbf30.png)
      ![image](https://user-images.githubusercontent.com/4283187/166292059-6cd8a595-f334-44d0-a8ef-d8fe2f3f6e80.png)
      ```
      Accuracy of Model: 0.8
      Test Accuracy of CNN 0.8
      ```

- Ensemble Model Test Result
  
  ```
  CM Logistic Regression Accuracy: 0.8
  CM Ada Boost Accuracy: 0.78
  CM Bagging Accuracy: 0.86
  ```

**Contributors**
- Pooja Goyal
- Roshan Sah
- Sanjib Paudel
- Sourabh Yadav
