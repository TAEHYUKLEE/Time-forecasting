#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import random
import sys

#Recursive 기본 setting을 늘려준다
sys.setrecursionlimit(3000)  

def DTW_matrix(A,B):
    
    A_len = len(A)
    B_len = len(B)
    
    Matrix = np.zeros([B_len+1,A_len+1], dtype = np.int32)
        
    Matrix[0,1:] = A[:]; Matrix[1:,0] = B[:]
    
    #make boundary condition
    Matrix[1,1] = abs(Matrix[0,1] -Matrix[1,0])
    
    for j in range(1,A_len):
        #A_boundary[i] = abs(A[i] - B[0]) + A_boundary[i-1]
        Matrix[1,j+1] = abs(Matrix[0,j+1] - Matrix[1,0]) + Matrix[1,j]
    for i in range(1,B_len):
        Matrix[i+1,1] = abs(Matrix[i+1,0] - Matrix[0,1]) + Matrix[i,1] 
        
    return Matrix

        
        
def DTW(A_len, B_len, Matrix,i):
    #print(Matrix.shape)
    
    if Matrix[B_len-1, A_len-1] != 0:
        return
    
    
    for j in range(2, A_len):
        Matrix[i,j] = abs(Matrix[i,0] - Matrix[0,j]) + min(Matrix[i-1,j],Matrix[i-1,j-1],Matrix[i,j-1])
        
    #print("hi")
    #print(Matrix)
    
    i+=1
    
    DTW(A_len, B_len, Matrix,i)

    
def Back_track(n,m, Matrix):
    n, m = n - 1, m - 1
    path = []


    while (m, n) != (0, 0):
        path.append((m, n))
        m, n = min((m - 1, n), (m, n - 1), (m - 1, n - 1), key=lambda x: Matrix[x[0], x[1]])
        #key가 여기서는 list = [(m - 1, n), (m, n - 1), (m - 1, n - 1)] 가 들어가고 lambda에 의해 계산이 되어 나온 것들로
        # min함수가 씌어지고 거기에 해당되는 key를 m,n으로 return해준다.

    path.append((0, 0))
    
    return path


def DTW_operation(A,B):
    
    A_len = len(A)+1; B_len = len(B)+1
    
    
    Matrix = DTW_matrix(A,B)
    
    #print(Matrix)
    
    DTW(A_len, B_len, Matrix,i=2)
    
    warp_path = Back_track(A_len, B_len,Matrix)
    
    print(Matrix)
    print(warp_path)    


    
def main():
    
    #A = np.arange(2000)
    #B = np.arange(2000)
    
    #random.shuffle(A)
    #random.shuffle(B)
    
    A = np.array([1,6,2,3,0,9,4,3,6], dtype = np.int32)
    B = np.array([1,3,4,9,8,2,1,5], dtype = np.int32)


    DTW_operation(A,B)
    

    
    
    
if __name__ == "__main__":
    main()



# In[100]:


import numpy as np 
n=3;m=3
i=0

#dictionary
d=lambda x, y: abs(x - y)
cost = np.array([[2,3,1,3,4],[4,1,6,4,6],[7,2,6,9,8],[7,5,2,1,6],[4,6,16,3,2]])
print(cost.shape)

print(d(2,3))

n, m = 4, 4
path = []

while (m, n) != (0, 0):
    path.append((m, n))
    
    path_p=[(m - 1, n), (m, n - 1), (m - 1, n - 1)]
    
    m, n = min(path_p, key=lambda x: cost[x[0], x[1]])

    #key=[(m - 1, n), (m, n - 1), (m - 1, n - 1)]
    #key(x):


path_p=[(m - 1, n), (m, n - 1), (m - 1, n - 1)]


stack = np.empty(0)

for i in range(2):
    key = path_p[i]
    value = cost[key[0],key[1]]
    
    A = [key, value]
    A = np.array(A)
    
    print(A)
    
    np.append(stack, A, axis=0)
    
print(stack)
    
    
    


#b,c=min((3-1,4),lambda x: cost[x[0], x[1]])    


print(lambda x: x[0]**2+x[1]**2(2,3))

path.append((0, 0))
    
print(path)

#m,n = min((1, 2), key=lambda x: cost[x[0], x[1]])

'''
while (n,m)!= (0,0):
    print(i)
    i +=1
'''


    
    


# In[17]:


import numpy as np
import random
import sys

sys.setrecursionlimit(2000)  

count = 0

def Distance_matrix(A,B):
    
    M = len(A)
    N = len(B)
    
    #For ED matrix
    D = np.zeros(shape=(M,N))
    
    for m in range(M):
        for n in range(N):
            D[m,n] = abs(A[m] - B[n])
            
    return D


def DTW_matrix(A,B):
    
    global count
    
    A_len = len(A)
    B_len = len(B)
    
    Matrix = np.zeros([B_len+1,A_len+1], dtype = np.int32)
        
    Matrix[0,1:] = A[:]; Matrix[1:,0] = B[:]
    
    #make boundary condition
    Matrix[1,1] = abs(Matrix[0,1] -Matrix[1,0])
    
    for j in range(1,A_len):
        #A_boundary[i] = abs(A[i] - B[0]) + A_boundary[i-1]
        Matrix[1,j+1] = abs(Matrix[0,j+1] - Matrix[1,0]) + Matrix[1,j]
        
        count +=1
        
    for i in range(1,B_len):
        Matrix[i+1,1] = abs(Matrix[i+1,0] - Matrix[0,1]) + Matrix[i,1] 
        
        count +=1
        
    return Matrix

        
        
def DTW(A_len, B_len, Matrix):
    
    global count
    #print(Matrix.shape)
    
    if Matrix[B_len-1, A_len-1] != 0:
        return
    
    for i in range(2, B_len):
        for j in range(2, A_len):
            Matrix[i,j] = abs(Matrix[i,0] - Matrix[0,j]) + min(Matrix[i-1,j],Matrix[i-1,j-1],Matrix[i,j-1])
            
            count +=1
            
    return Matrix[B_len-1, A_len-1]
        

    
def Back_track(n,m, Matrix):
    
    global count
    
    n, m = n - 1, m - 1
    path = []


    while (m, n) != (0, 0):
        path.append((m, n))
        m, n = min((m - 1, n), (m, n - 1), (m - 1, n - 1), key=lambda x: Matrix[x[0], x[1]])
        #key가 여기서는 list = [(m - 1, n), (m, n - 1), (m - 1, n - 1)] 가 들어가고 lambda에 의해 계산이 되어 나온 것들로
        # min함수가 씌어지고 거기에 해당되는 key를 m,n으로 return해준다.
        
        count +=1

    path.append((0, 0))
    
    return path


def DTW_operation(A,B):
    
    A_len = len(A)+1; B_len = len(B)+1
    
    ED_matrix = Distance_matrix(A,B)
    
    Matrix = DTW_matrix(A,B)
    
    #print(Matrix)
    
    distance = DTW(A_len, B_len, Matrix)
    
    warp_path = Back_track(A_len, B_len,Matrix)

    
    print("DP DTW-distance:", distance)
    print(distance)
    print("\nDTW-Matrix")
    print(Matrix)
    print("\nWarp-path")
    #print(warp_path)
    #print(np.array(warp_path).shape)


    
def main():
    
    A = np.arange(9000)
    B = np.arange(9000)
    
    random.shuffle(A)
    random.shuffle(B)
    
    #A = np.array([1,6,2,3,0,9,4,3,6], dtype = np.int32)
    #B = np.array([1,3,4,9,8,2,1,5], dtype = np.int32)


    DTW_operation(A,B)
    

    
    
    
if __name__ == "__main__":
    
    main()
    
    print("\nComplexity")
    print(count)


# In[7]:


#brute force algorithm for DTW (M+N)!/N!M! 이라서 난 3^n이라고 생각하는데
#DP법과 같이 똑같이 Matrix를 채워야 하긴 함 근데 그것을 Recursive relation이 아닌 다 해보는거.

import numpy as np
import random
import sys 

count=0


def Distance_matrix(A,B):
    
    M = len(A); N = len(B)
    
    #For ED matrix
    D = np.zeros(shape=(M,N))
    
    for i in range(M):
        for j in range(N):
            D[i,j] = abs(A[i] - B[j])
            
    return D,M,N


#이런 방식으로 Function화 시켜서 한꺼번에 돌리는 부분
def get_path(M,N):
  
    def get_direction(i,j):
        temp = []
        #for next step
        if i != M-1 and j != N-1 : 
            temp.append((i,j+1))
            temp.append((i+1,j))
            temp.append((i+1,j+1))
        #맨 오른쪽에 막혔을때 위로 올라가게 한다 (to N,M)
        elif i == M-1 and j != N-1 : 
            temp.append((i,j+1))
        #맨 위에 막혔을때 오른쪽으로 가게 한다 (to N,M)
        elif i != M-1 and j == N-1 : 
            temp.append((i+1,j))
            
        return temp
    

    def recursive_explore(step,previous_step=[]):
        i,j = step
        current_step = previous_step[:] 
        current_step.append((i,j)) #현재 Position
        next_step = get_direction(i,j)
   
        if len(next_step)==0 : 
            #Final state --> 위의 get_direction부분에서 더 이상 첨가되지 않고 넘어옴
            all_paths.append(current_step) #현재 지점을 마지막으로 해서 저장
        else:
            #경로가 끝나지 않았을때 Recursive하게 더 파고 들어가서 끝까지 경로를 끝내온다.
            for _ in next_step:
                recursive_explore(_, previous_step = current_step)
        #하나의 경로가 다 마무리가 되면 다시 recursive로 돌아와서 시작한다 그래서 마지막부분에서 3개가 마무리됨.
  
    all_paths = [] #Final paths([path1], [path2],[path3],[path4],[path5]......)
    #from (0,0) step부터 시작한다
    Initial_step = (0,0)
    recursive_explore(Initial_step,previous_step=[])

    return all_paths

#앞서 구한 Path들에 대해 각각 Euclidean distance를 구하는 부분
def all_distance(D_matrix,path):
    
    global count
    
    dtw_distance = 0
    for (i,j) in path: 
        dtw_distance += D_matrix[i,j]
        count +=1
        
    return dtw_distance


#최종적으로 구한 path와 dtw_distance중 최솟값을 반환하도록 한다.
def get_dtw_distance(D_matrix, all_paths):
    all_cases = [all_distance(D_matrix, path) for path in all_paths]
    return min((v,i) for i,v in enumerate(all_cases))
    
    
    
def main():
    
    A = np.arange(12)
    B = np.arange(12)
    
    random.shuffle(A)
    random.shuffle(B)
    
    #A = np.array([1,6,2,3,0,9,4,3], dtype = np.int32)
    #B = np.array([1,3,4,9,8,2,1,5], dtype = np.int32)


    D_matrix,M,N = Distance_matrix(A,B)
    all_paths = get_path(M,N)
    dtw_distance,index = get_dtw_distance(D_matrix, all_paths)
    
    print("Euclidean Distance Matrix")
    print(D_matrix)
    print("\nNumber of paths (Complexity)")
    print(len(all_paths))
    print("\nDP DTW-distance:", dtw_distance)
    print(dtw_distance)
    print("\nWarp_path")
    print(all_paths[index][:9],all_paths[index][9:])
    print("\nComplexity")
    print(count)
    
if __name__ == "__main__":
    
    main()



# In[ ]:


print("Euclidean Distance Matrix\n" , D_matrix)
print("Number of paths (Complexity)\n"   , len(all_paths))
print("DTW-distance\n"   , optimal_cost)
print("Warp_path\n"         ,all_paths[idx][:9])
print("               "     ,all_paths[idx][9:])


# In[6]:


#brute force algorithm for DTW (M+N)!/N!M! 이라서 난 3^n이라고 생각하는데
#DP법과 같이 똑같이 Matrix를 채워야 하긴 함 근데 그것을 Recursive relation이 아닌 다 해보는거.

import numpy as np
import random
import sys 

count=0


def Distance_matrix(A,B):
    
    M = len(A); N = len(B)
    
    #For ED matrix
    D = np.zeros(shape=(M,N))
    
    for i in range(M):
        for j in range(N):
            D[i,j] = abs(A[i] - B[j])
            
    return D,M,N


#이런 방식으로 Function화 시켜서 한꺼번에 돌리는 부분
def get_path(M,N):
  
    def get_direction(i,j):
        temp = []
        #for next step
        if i != M-1 and j != N-1 : 
            temp.append((i,j+1))
            temp.append((i+1,j))
            temp.append((i+1,j+1))
        #맨 오른쪽에 막혔을때 위로 올라가게 한다 (to N,M)
        elif i == M-1 and j != N-1 : 
            temp.append((i,j+1))
        #맨 위에 막혔을때 오른쪽으로 가게 한다 (to N,M)
        elif i != M-1 and j == N-1 : 
            temp.append((i+1,j))
            
        return temp
    

    def recursive_explore(step,previous_step=[]):
        i,j = step
        current_step = previous_step[:] 
        current_step.append((i,j)) #현재 Position
        next_step = get_direction(i,j)
   
        if len(next_step)==0 : 
            #Final state --> 위의 get_direction부분에서 더 이상 첨가되지 않고 넘어옴
            all_paths.append(current_step) #현재 지점을 마지막으로 해서 저장
        else:
            #경로가 끝나지 않았을때 Recursive하게 더 파고 들어가서 끝까지 경로를 끝내온다.
            for _ in next_step:
                recursive_explore(_, previous_step = current_step)
        #하나의 경로가 다 마무리가 되면 다시 recursive로 돌아와서 시작한다 그래서 마지막부분에서 3개가 마무리됨.
  
    all_paths = [] #Final paths([path1], [path2],[path3],[path4],[path5]......)
    #from (0,0) step부터 시작한다
    Initial_step = (0,0)
    recursive_explore(Initial_step,previous_step=[])

    return all_paths

#앞서 구한 Path들에 대해 각각 Euclidean distance를 구하는 부분
def all_distance(D_matrix,path):
    
    global count
    
    dtw_distance = 0
    for (i,j) in path: 
        dtw_distance += D_matrix[i,j]
        count +=1
        
    return dtw_distance


#최종적으로 구한 path와 dtw_distance중 최솟값을 반환하도록 한다.
def get_dtw_distance(D_matrix, all_paths):
    all_cases = [all_distance(D_matrix, path) for path in all_paths]
    return min((v,i) for i,v in enumerate(all_cases))
    
    
    
def main():
    
    #A = np.arange(12)
    #B = np.arange(12)
    
    #random.shuffle(A)
    #random.shuffle(B)
    
    A = np.array([1,6,2,3,0,9,4,3,6], dtype = np.int32)
    B = np.array([1,3,4,9,8,2,1,5], dtype = np.int32)


    D_matrix,M,N = Distance_matrix(A,B)
    all_paths = get_path(M,N)
    dtw_distance,index = get_dtw_distance(D_matrix, all_paths)
    
    print("Euclidean Distance Matrix")
    print(D_matrix)
    print("\nNumber of paths (Complexity)")
    print(len(all_paths))
    print("\nBrute force DTW-distance:", dtw_distance)
    print(dtw_distance)
    print("\nWarp_path")
    print(all_paths[index][:9],all_paths[index][9:])
    print("\nComplexity")
    print(count)
    
if __name__ == "__main__":
    
    main()


