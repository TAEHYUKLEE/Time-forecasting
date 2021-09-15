
import numpy as np 
import pandas as pd # to read the dataset as dataframe
import random #for the random.shuffle function
import tensorflow as tf
import csv 

#앞서 Training set 70%, Validation set 20%, Test set 10%로 데이터를 분할하였다.

##########################Training set을 pandas로 불러들인다 #############################
data= pd.read_csv("C:\Read\Training\Training set.csv", encoding = "ISO-8859-1")

#데이터 frame의 Dimension의 수를 프린트 해준다 M by N (M, N표기)
#print(np.shape(data))

#for the training set
#data.iloc[0:65860,0:18]

number= (data.shape[0]-85+1) #number of batch elements


# 위에서 읽어들인 Dataset을 처리하는 과정 (Validation set)
#print(np.shape(data))
#print(data.iloc[:,4:]) #pandas로 읽어들인 데이터를 print할경우 row와 column을 알려준다.

#Training dataset slice
data1=data.iloc[:,4:] #사용할 데이터만 슬라이스하여 data1로 선언하였다.


### numpy로 하기 위해서는 다음과 같이 한다 ######
#Trainin dataset
data2 = data1.to_numpy() #Pandas로 읽어들인 Training데이터를 numpy로 전환하여 numpy module을 이용한다.

########################## Validation dataset을 불러들여온다 ##############################
#Validation set을 pandas로 불러들인다.
val_data = pd.read_csv("C:\Read\Validation\Validation set.csv",encoding = "ISO-8859-1")

#Validation dataset slice
val1=val_data.iloc[:,4:] #Validation error를 구하기 위해 사용할 data를 슬라이스하여 val1로 저장한다.

#Validation dataset
val2 = val1.to_numpy() # Pandas로 읽어들인 Validation데이터를 numpy로 전환하여 numpy module로 이용하도록 한다

##############################################################################################

########################## Train batch 형성 def ###############################################
def get_batch(batch_x, batch_y, data2, index):
    for j in range(batch_size):
        batch_x[j:j+1,:] = np.reshape(data2[index[j]   :index[j]+72,0 :14], (1,72*14)) 
        #Index가 임의로 추출됨 정수로 (바꿔야함 pm2.5포함-->이유 현재의 PM2.5는 내가 알고 있고 이걸 통해서 미래의 PM2.5를 예측하는거)
        batch_y[j:j+1,:] = np.reshape(data2[index[j]+84:index[j]+85,13:14], (1,1))
    return batch_x, batch_y


# Hyperparameters
learning_rate = 0.0001
training_epochs = 50
batch_size = 100 # Batch size 찾아볼 것 (100개의 elements를 하나의 batch로 정의한다)
beta = 0.1
mean=23.95
std=15.94047


batch_x = np.zeros((batch_size,72*14)) 
batch_y = np.zeros((batch_size,1)) 

index_array=np.arange(data.shape[0]-85+1) # training data batch elements

# Network Parameters
n_hidden_1 = 16 # 1st layer number of neurons: Number of Nodes
n_hidden_2 = 16 # 2nd layer number of neurons: Number of Nodes
n_hidden_3 = 16
n_hidden_4 = 16
n_input =  14*72 #어떤 형식인지 알아야 한다. vector화 시켜야 하긴 하는데
n_output = 1 # predicted total classes --> Predicted number batch의 개수만큼

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])

w1= tf.get_variable("w1", shape=[n_input, n_hidden_1],initializer=tf.contrib.layers.variance_scaling_initializer()) #해보니까 Tensorflow 2.0은 tf.contrib module이 포함되어 있지 않다.
w2= tf.get_variable("w2", shape=[n_hidden_1, n_hidden_2],initializer=tf.contrib.layers.variance_scaling_initializer())  #w1에서 오는데이터 형태를 맞춰서 w2에 맞춰줘야한다.
w3= tf.get_variable("w3", shape=[n_hidden_2, n_hidden_3],initializer=tf.contrib.layers.variance_scaling_initializer())
w4= tf.get_variable("w4", shape=[n_hidden_3, n_hidden_4],initializer=tf.contrib.layers.variance_scaling_initializer())
w_out= tf.Variable(tf.random_normal([n_hidden_4, n_output]))

b1 = tf.Variable(tf.zeros([n_hidden_1]))
b2 =  tf.Variable(tf.zeros([n_hidden_2]))
b3 =  tf.Variable(tf.zeros([n_hidden_3]))
b4 =  tf.Variable(tf.zeros([n_hidden_4]))
b_out= tf.Variable(tf.zeros([n_output]))


# Create model (2 numbers of layers with 256 nuerons) Neural network 아키텍쳐
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    linear_1 = tf.add(tf.matmul(x, w1), b1)
    layer_1= tf.nn.relu(linear_1) #activation function으로 부분선형함수인 relu fucntion을 집어넣었다.
    
    #Activation function을 집어넣어줘 봐야한다 (근데 왜? 활성함수를?) --> 정확한 메커니즘 알아보기
    
    # Hidden fully connected layer with 256 neurons
    linear_2 = tf.add(tf.matmul(layer_1, w2), b2)
    layer_2= tf.nn.relu(linear_2)

    linear_3 = tf.add(tf.matmul(layer_2, w3), b3)
    layer_3= tf.nn.relu(linear_3)

    linear_4 = tf.add(tf.matmul(layer_3, w4), b4)
    layer_4= tf.nn.relu(linear_3)
    
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_4, w_out) + b_out
    return out_layer

#착가했던게 Y를 직접 넣어서 W를 조저해주는게 아니라 Loss function을 줄여서 w를 학습시키는 것이다.

######################### Optimization of Loss function ###########################
Y_pred = multilayer_perceptron(X)
       
# Define loss and optimizer

# Loss part in function (Mean square of Error)
loss = tf.reduce_mean((Y_pred-Y)**2) #신경회를 거쳐 나온 Y_pred와 실제 Y와의 Loss를 줄여야 한다. (이걸로 학습시키는 것)
#Y_pred안에 w가 들어가 있으니까 w를 학습하게 되는 것이다. 그래서 Regulazier에서도 w1, w2, w_out에 대한 정규화를 해주는 것.

# Regulazier (L2)
regularizer = tf.nn.l2_loss(w1)+tf.nn.l2_loss(w2)+tf.nn.l2_loss(w_out) #왜 3가지에 대해서 더하게 되는건지?

#Optimizer AdamOptimizer (kind of Descent gradient method)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) #AdamOptimizer

#Loss function (Loss + Regulazation)
loss_op1 = loss #loss의 평균 --> Mean sq
loss_op2 = beta * regularizer
loss_op = loss_op1 + loss_op2
#Perform that method
train_op = optimizer.minimize(loss_op) #optimizer.minimize 앞에서 설정한 Optimizer로 Minimize를 수행한다.

####################################################################################

saver = tf.train.Saver(max_to_keep=None)
sess =  tf.Session() #tf.Seesion() 실행을 간단히 sess라고 간략화시킨다.
sess.run(tf.global_variables_initializer()) # 모든 Tensor들을 초기화시킨다

'''
Restorefile = 'C:\Write\1\MODEL.ckpt-1'
saver.restore(sess, Restorefile)
'''

outfile = open("Cost_Validation 12hrs.txt", 'w')

# Training cycle
for epoch in range(training_epochs):
    avg_cost1 = 0
    total_batch = int((data.shape[0]-85+1)/batch_size) # Total batch의 개수는 전체 elements의 개수를 내가 구성하고자 하는 batch의 size로 나눈것. int 정수
    random.shuffle(index_array) # epoch1번 반복할때마다 shuffle돼서 새로운 batch를 뽑아낸다 

    # Loop over all batches
    for i in range(total_batch):
                        
        ########################### Slice batch matrix into input matrix & output matrix ###############################

        batch_x, batch_y = get_batch(batch_x, batch_y, data2, index_array[(i)*batch_size:(i+1)*batch_size])
        #get_batch함수를 불러온다. batch_x, batch_y를 100개의 batch elements로 형성해서 return하여 돌아오도록한다. 
        # x_batch = 72시간 데이터를 (14개 이용)  --> y_batch= 24시간 후의 pm2.5를 예측
        ####################################################################################################
        
                 
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c1 = sess.run([train_op, loss_op1], feed_dict={X: batch_x,Y: batch_y})
        #sess.run (node와 선으로 짜여진 graph) train_op, loss_op1을 실행시켜 각 _, c1에 저장한다.
            
        # Compute average loss
        avg_cost1 += c1 / total_batch 
        #아 total batch만큼 나누는게 평균이? 어차피 안에 들어가 있는 for loop는 total_batch 658번하는 다 하는거니까 
        #  number frist loop/total_batch +  second loop/total_batch + ..... + 658 th loop/totla_batch = 전체 cost의 평균이된다.

        ################################# batch 한 번 Training 돌리는 구간 ###################################
    
    index_val_array = np.arange(val2.shape[0]-85+1)
    number_batch_elements=val2.shape[0]-85+1
    #validation의  totla number of batch elements

    #batch_val_x, batch_val_y를 다시 정의해준다. 
    batch_val_x = np.zeros((val2.shape[0]-85+1,72*14))
    batch_val_y = np.zeros((val2.shape[0]-85+1,1))


    ################ get_val_batch함수 batch를 형성시킬수 있는 함수module을 만든다 ###############
    def get_val_batch(batch_val_x, batch_val_y, val2, index):
        for j in range(val2.shape[0]-85+1):
            batch_val_x[j:j+1,:] = np.reshape(val2[index[j]   :index[j]+72,0 :14], (1,72*14)) 
            batch_val_y[j:j+1,:] = np.reshape(val2[index[j]+84:index[j]+85,13:14], (1,1))
        return batch_val_x, batch_val_y
    #total batch elements를 하나의 batch로 하여 뽑아낸다.
    ##############################################################################################


    ##########get_val_batch함수로 가서 validation set에 대한 batch를 만들어 온다 ########

    #처음에 val2.shape[0]-96+1하면 하나의 숫자만 나온다 그런데 그 아래 0:batch_size는 하나의 숫자가 아니라 나열

    batch_val_x, batch_val_y = get_val_batch(batch_val_x, batch_val_y, val2, index_val_array)
    #print(np.shape(batch_val_x))
    #print(np.shape(batch_val_y))
    #####################################################################################


    #이미 위에 multilayer_perceptron은 지역함수로 정의해놨다#

    n_val_input =  14*72 
    n_val_output = 1 
    avg_val1 =0

    # tf Graph input
    X_val = tf.placeholder("float", [None, n_val_input])
    Y_val = tf.placeholder("float", [None, n_val_output])

    Y_val_pred = multilayer_perceptron(X_val)

    predicted_val_value, real_val_value = sess.run([Y_val_pred,Y_val], feed_dict={X_val: batch_val_x,Y_val: batch_val_y})

    
    #Validation error를 계산해낸다.

    avg_val1 = np.mean((predicted_val_value[:number_batch_elements] - real_val_value[:number_batch_elements])**2)

    # Display logs per epoch step
    if epoch % 1 == 0: #epoch 단계마다 보겠다 1마다 보겠다라는 것을 의미함.
        weight_loss = sess.run(loss_op2) #loss_op2 beta* regularizer 하위 node부터 계산해서 위의 계산값을 보겠다.
          
        #c1 = sess.run(loss_op1, feed_dict={X: batch_x,Y: batch_y})
            
        print("Epoch:", '%04d' % (epoch+1), "Training_cost={:.9f}".format(avg_cost1), "Regular_cost={:.9f}".format(weight_loss))
        print(f"Valdiation_error: {avg_val1}")   
        
    ### Write Cost & Validation ###
    outfile.write(str(epoch+1) + "     " + str(avg_cost1) + "      " + str(avg_val1) + "\n")
outfile.close()
        
print("Training Finished!")
        
    
##########################Test set을 pandas로 불러들인다 #############################
test= pd.read_csv("C:\Read\Test\Test set.csv", encoding = "ISO-8859-1")

# test dataset 확인
number= (test.shape[0]-85+1) #number of batch elements

#Test dataset slice
test1=test.iloc[:,4:] #사용할 데이터만 슬라이스하여 data1로 선언하였다.

### numpy로 하기 위해서는 다음과 같이 한다 ######
#Trainin dataset
test2 = test1.to_numpy() #Pandas로 읽어들인 Training데이터를 numpy로 전환하여 numpy module을 이용한다.


########################## Test batch 형성 함수 ###############################################
def get_test_batch(batch_test_x, batch_test_y, test2, index):
    for j in range(test2.shape[0]-85+1):
        batch_test_x[j:j+1,:] = np.reshape(test2[index[j]   :index[j]+72,0 :14], (1,72*14)) 
        batch_test_y[j:j+1,:] = np.reshape(test2[index[j]+84:index[j]+85,13:14], (1,1))
    return batch_test_x, batch_test_y
###############################################################################################


# Hyperparameters

number_test_elements = test2.shape[0]-85+1 # Batch size 찾아볼 것 (100개의 elements를 하나의 batch로 정의한다)

batch_test_x = np.zeros((test2.shape[0]-85+1,72*14))
batch_test_y = np.zeros((test2.shape[0]-85+1,1))

index_test_array=np.arange(test2.shape[0]-85+1) # 미쳤다 Data.test2[0]이 아님


# tf Graph input
X_test = tf.placeholder("float", [None, n_input])
Y_test = tf.placeholder("float", [None, n_output])


batch_test_x, batch_test_y = get_test_batch(batch_test_x, batch_test_y, test2, index_test_array)
# 여기까지는 문제가 없다.

Y_test_pred = multilayer_perceptron(X_test)

####################################################################################


predicted_test_value, real_test_value = sess.run([Y_test_pred,Y_test], feed_dict={X_test: batch_test_x,Y_test: batch_test_y})

accuracy_percent = predicted_test_value/real_test_value*100

print(f"predicted_test_value: {predicted_test_value}")
print(f"real_test_value: {real_test_value}")  #내가 real_value_print를 predicted_test_value로 했었다. 그래서 이상하게 값이..
print(f"Accuracy_percent(%):{accuracy_percent}")

avg_test1 = np.mean((predicted_test_value[:number_test_elements] - real_test_value[:number_test_elements])**2)

print(f"Accuracy_error: {avg_test1}")

#################### Write test result as csv file ################
a= predicted_test_value*std+mean
b= real_test_value*std+mean
c= abs(a-b)
d=np.concatenate((a,b,c),axis=1)

np.savetxt("test result 24hrs.csv", d, delimiter=",")

###################################################################
