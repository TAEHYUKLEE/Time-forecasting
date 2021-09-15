#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd

#(Dataframe)
def norm(d_f):
    
    CL = d_f.iloc[:, 1:].columns  
    
    d_a= np.array(d_f)
    d_a = d_a[:,1:] # 첫 열은 삭제    
        
    R_n=d_a.shape[0] #Observation 개수 입력 (row #)
    C_n=d_a.shape[1] #변수 개수 입력 (Columns #)   
    
    sum_1=np.zeros([C_n], dtype=np.float64)
    square=np.zeros([C_n], dtype=np.float64)
    mean=np.zeros([C_n], dtype=np.float64)
    std = np.zeros([C_n], dtype=np.float64)
    norm = np.zeros([R_n,C_n], dtype=np.float64)   

   

    
    R_n=d_a.shape[0] #Observation 개수 입력 (row #)
    C_n=d_a.shape[1] #변수 개수 입력 (Columns #)

 

    #################### Summation loop ####################
    for i in range (R_n):   
        for j in range (C_n):
            sum_1[j] = sum_1[j] + d_a[i,j]
    
    #################### Average ################### 
    mean=sum_1/R_n
    
    #################### Standard deviation ###################
    for j in range (C_n):
        for i in range (R_n):
            square[j] = np.square(d_a[i,j] - mean[j]) + square[j]
    
    square= square/(R_n-1)  
    std = np.sqrt(square)


    ####################### Data normalized #####################  
       #Plot graph with normalize (plot normalize data)
    
    for j in range (C_n):
        for i in range (R_n):
            norm[i,j] = (d_a[i,j] - mean[j])/std[j]
        
    #print('\n"normalized dataset"')
    #print(norm.shape)
    #print(CL.shape)
    
    norm = pd.DataFrame(norm, columns = CL)
    norm.to_csv('normalized_dataset (_Yong sang).csv')
    
    print('\n"Saved normalized data as csv format"\n')
    
    return norm, R_n, C_n

