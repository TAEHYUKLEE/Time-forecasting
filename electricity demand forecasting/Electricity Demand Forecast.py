#!/usr/bin/env python
# coding: utf-8

# In[1]:


#용상 Project only NN code

import numpy as np 
import pandas as pd # to read the dataset as dataframe
import random #for the random.shuffle function
import tensorflow.compat.v1 as tf; tf.compat.v1.disable_eager_execution()
import normalization as norm # 직접 만든 normalization module
import csv 

df= pd.read_csv("C:\Read\Final dataset.csv", encoding = "ISO-8859-1") 

f = open('data file.csv', 'w', encoding='utf-8', newline='')

(norm, R, C) = norm.norm(df) #normalization module 만들어놓음

######### Split full dataset into training(70%), validation(20%), test(10%) ########
N_tr = R*7//10
N_va = R*2//10
N_te = R - (N_tr+ N_va)


train_fr = norm.iloc[0:N_tr, :]
val_fr = norm.iloc[N_tr:(N_va+N_tr), :]
test_fr = norm.iloc[(N_va+N_tr):(N_va+N_tr+N_te), :]


train_set = np.array(train_fr)
val_set = np.array(val_fr)
test_set = np.array(test_fr)

##############################################################################################




###########################    formation of Neural Netowrk  ################################# 

#Neural Network size parameter
n_input = (C-1) #C is numbers of variables
n_output =1
node_layer_1 = 128
node_layer_2 = 128
node_layer_3 = 128
batch_size = 100


#tensor들에 대한 Input output, variable들을 정의해준다 (define tensor)
X_input = tf.placeholder("float", [None, n_input]) #type tensor type (데이터 유형)
Y_real = tf.placeholder("float", [None, n_output]) #나중에 Loss에서 쓰일 y값

weight_1 = tf.Variable(tf.random_normal([n_input, node_layer_1]))
weight_2 = tf.Variable(tf.random_normal([node_layer_1, node_layer_2]))
weight_3 = tf.Variable(tf.random_normal([node_layer_2, node_layer_3]))
weight_out = tf.Variable(tf.random_normal([node_layer_3, n_output]))

bias_1 = tf.Variable(tf.random_normal([1, node_layer_1]), dtype=tf.float32)
bias_2 = tf.Variable(tf.random_normal([1, node_layer_2]), dtype=tf.float32)
bias_3 = tf.Variable(tf.random_normal([1, node_layer_3]), dtype=tf.float32)
bias_out = tf.Variable(tf.random_normal([n_output]))

saver = tf.train.Saver()
#shape은 numpy, tensor와 혼용해서 쓰지 못하네

# Formation of Neural Network 
def MLP (X_input):
    
    #모든 w_1 ~ w_N (Hidden layer) w_1 ~ w_N' (Hidden layer) 모든 경우의수의 조합을 Matrix로 나타내게 된 것
    linear_1 = tf.matmul(X_input, weight_1) + bias_1
    active_1 = tf.nn.sigmoid(linear_1) #sigmoid를 하든 tanh하든, relu를 하든 activation function은 알아서 정하기를
    #active matrix dimension [batch_size, node number_layer1]
    #weight_2 matrix dimension [node number_layer1, node_layer_2]
    
    linear_2 = tf.add(tf.matmul(active_1, weight_2), bias_2) #tf.add로 안하고 그냥 더하면 data type문제가 생김
    active_2 = tf.nn.sigmoid(linear_2)
    #Numpy는 Dataframe말고 array나 list 뿐만 아니라 Tensor까지 모두 빠르게 계산해준다.
    
    linear_3 = tf.matmul(active_2, weight_3) + bias_3
    active_3 = tf.nn.sigmoid(linear_3)
    
    Y_out = tf.matmul(active_3, weight_out) + bias_out 

    return Y_out

########### Define Loss function - learning part ##############
#learning parameter
learning_rate = 0.0001
B = 0.001

Y_p =  MLP(X_input)

MSE = tf.reduce_mean((Y_p - Y_real)**2)

#L2 regularization
regular = tf.nn.l2_loss(weight_1)+tf.nn.l2_loss(weight_2)+tf.nn.l2_loss(weight_out)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

Loss = MSE + B*regular

train = optimizer.minimize(Loss)  #앞에서 구한 Loss에 대해 AdamOptimizer로 train (Optimization) 시킨다

###################################################################

sess =  tf.Session() 
sess.run(tf.global_variables_initializer()) 
#모든 tensor의 variable들을 initialization할 수 있다. (딱 초기화까지 시켜놓고 끝냄)

# 위에서 Neural network에 필요한 size와 구성을 끝냈다. 불러온 데이터를 통해 Batch를 형성하고 넣어줘야한다


# Batch는 training 목적에 따라 형성 시키는것이 조금씩 달라진다.


#Batch Parameter
N_batch = train_set.shape[0]//batch_size
# 여기서 있는 그대로 학습을 시키기 위해 Random하게 Batchsize 100개로 Training epoch 개수 (전체 Instnace 개수 //100 개를 한다)

# Training batch
batch_X = np.zeros((batch_size,C-1)) #Batch 형태를 잡아줌
batch_Y = np.zeros((batch_size,1))

# Validation batch
batch_val_X = np.zeros((val_set.shape[0],C-1)) #Batch 형태를 잡아줌
batch_val_Y = np.zeros((val_set.shape[0],1))

# Test batch
batch_test_X = np.zeros((test_set.shape[0],C-1)) #Batch 형태를 잡아줌
batch_test_Y = np.zeros((test_set.shape[0],1))

#print(batch_X.shape)
#print(train_set.shape)


def formation_batch(batch_X, batch_Y, train_set, index, batch_size):
    for i in range(batch_size):
       # print(index.shape)
        batch_X[i:i+1,:] = train_set[index[i]:index[i]+1,1:C]
        #index[i]:index[i+1] -- 이건 왜 안될까? index[i]:index[i]+1로 해야하네
        #ValueError: could not broadcast input array from shape (0,38) into shape (1,38)
        batch_Y[i:i+1,:] = train_set[index[i]:index[i]+1,0:1] #여기서 위에서 나눠 놓은 Train set을 이용한다
    return batch_X, batch_Y    


#######################################################################################################

path = tf.train.latest_checkpoint("saved")
print(path)


    
# Training part    
epoch_num = 50

#Shuffle 시킬 index
index=np.arange(train_set.shape[0]) 
#index에서 batch_size를 빼야하지 않나? 왜 안빼니까 되는거지?
#한 Instance 가 하나의 batch 요소로 들어가기때문에 train_set row number에서 무언가 따로 뺄 필요가 없다
#착각함... Batch자체는 여러 행에서 무작위로 뽑아다 형성시키는거였음
#print(index)
#print(N_batch)

#만약 index가 R이면 그 이상의 index가 없다


f.write('"epoch", "Cost", "Val_error", "L2_error" \n')

for epoch in range(epoch_num):
    cost_mean = 0
    random.shuffle(index)  
       
    for i in range(N_batch):
                        
                                     ############# Training part ##############
       #Batch를 만들어서 가져온다
        batch_X, batch_Y = formation_batch(batch_X, batch_Y, train_set, index[(i)*batch_size:(i+1)*batch_size], batch_size)        
                 
        #학습시키는 구문
        #batch_X에서는 Y_pred을 뽑아내고, Y_real에서는 진짜 Y_real값을 가져온다
        # sess.run(train, feed_dict={X_input: batch_X,Y_real: batch_Y})
        #B = sess.run(MSE, feed_dict={X_input: batch_X,Y_real: batch_Y})
        
        _, cost = sess.run([train, MSE], feed_dict={X_input: batch_X,Y_real: batch_Y})
        # Compute average loss
        cost_mean += cost /N_batch   
    
    #위에서 하나의 Batch로 학습을 시킨 [weight, bias]를 가지고 Validation을 해 보자.  

    
                                     ############# Validation part ################
        
    #위에서 한나의 Batch로 학습시킨 것에 대해서 Validation해보자
    Val_error =0
    index_val = np.arange(val_set.shape[0]) 
    N_val = len(index_val) #index_val 개수를 채워줘야한다
    #print(N_val)
    #Batch를 채워와야 한다. Validation batch는 training batch와 size가 다르니까 새로 선언해줘야한다

    batch_val_X, batch_val_Y = formation_batch(batch_val_X, batch_val_Y, val_set, index_val, N_val)
    ##### Validation set data하고 빈 batch_val_X하고 Index하고 같이 보내줘서 validation batch를 만들어 온다

    Y_val_op = MLP(X_input)
    #앞에 만들어놓은 MLP를 그대로 이용 바로 위에서 만들어온 VALIDATION Batch를 넣을 Neural network를 다시 만들어준다
    

    val_pre, val_real = sess.run([Y_val_op,Y_real], feed_dict={X_input: batch_val_X, Y_real: batch_val_Y})
    #Feed로 앞에서 만들어온 batch_val을 넣어서 feeding해준다 Y_val_op (MLP) 와 Y_real에 넣어준다

    
    #Validation error를 계산해낸다.

    Val_error = np.mean((val_pre[:] - val_real[:])**2)
    #여기서 tf.reduce_mean을 쓰면 편하지만, sess.run()을 한 번 더 해줘야 하니까 #np.mean
    
    # Display logs per epoch step
    if epoch % 1 == 0: #epoch 단계마다 보겠다 1마다 보겠다라는 것을 의미함.
        L2_error = sess.run(regular) #loss_op2 beta* regularizer 하위 node부터 계산해서 위의 계산값을 보겠다.
        
        print('epoch:"%d"' % (epoch+1))
        print("Cost = %f" %cost_mean, "Valdiation_error = %f" %Val_error,"L2_error = %f" %L2_error)
        
    ### Write Cost & Validation ###
    f.write("%d, %f, %f, %f \n" %(epoch+1, cost_mean, Val_error, L2_error))
    
    ckpt_path = saver.save(sess, "save_file/train1", epoch)
        
f.close()    

print("Finished!!!\n")
print("\nLet's start testing the learned model")
    
                                ################### Test part #####################
test_error =0
index_test = np.arange(test_set.shape[0])
N_test = len(index_test) #index_val 개수를 채워줘야한다

batch_test_X, batch_test_Y = formation_batch(batch_test_X, batch_test_Y, test_set, index_val, N_test)
##### test set data하고 빈 batch_test_X하고 Index하고 같이 보내줘서 test batch를 만들어 온다

Y_test_op = MLP(X_input)
#앞에 만들어놓은 MLP를 그대로 이용 바로 위에서 만들어온 test Batch를 넣을 Neural network를 다시 만들어준다   

test_pre, test_real = sess.run([Y_test_op,Y_real], feed_dict={X_input: batch_test_X, Y_real: batch_test_Y})
#Feed로 앞에서 만들어온 batch_test을 넣어서 feeding해준다 Y_test_op (MLP) 와 Y_real에 넣어준다
    
#test error를 계산한다

test_error = np.mean((test_pre[:] - test_real[:])**2)
#여기서 tf.reduce_mean을 쓰면 편하지만, sess.run()을 한 번 더 해줘야 하니까 #np.mean

print("Accuracy error = %f" %test_error)

