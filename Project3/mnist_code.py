# -*- coding: utf-8 -*-
"""
Created on Fri May  6 21:29:18 2016

@author: abhitrip
"""


import numpy as np
import sklearn
import scipy as sc
import os
import gzip
import struct

#%%
# transfer functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# derivative of sigmoid
def dsigmoid(y):
    return y * (1.0 - y)

# using softmax as output layer is recommended for classification where outputs are mutually exclusive
def softmax(w):
    e = np.exp(w - np.amax(w))
    dist = e / np.sum(e)
    return dist

# using tanh over logistic sigmoid for the hidden layer is recommended   
def tanh(x):
    return np.tanh(x)
    
# derivative for tanh sigmoid
def dtanh(y):
    return 1 - y*y

def extract_file(filename):
    out_name = os.path.splitext(filename)[0]
    out_file = open(out_name,'wb')
    try:
        with gzip.open(filename,'rb') as fp:
            data = fp.read()
            out_file.write(data)
        
    except:
        print 'File %s could not be extracted. Please check \n' % filename
    out_file.close()
    statinfo = os.stat(out_name)
    print statinfo.st_size
    print 'File %s extracted to %s with size =%d' %(filename,out_name,statinfo.st_size)
    return out_name
train_data_file = extract_file('train-images-idx3-ubyte.gz')
test_lab_file   = extract_file('t10k-labels-idx1-ubyte.gz')
test_data_file  = extract_file('t10k-images-idx3-ubyte.gz')
train_lab_file  = extract_file('train-labels-idx1-ubyte.gz')

#%%

#def correct_endianness(filename,file_type
def read_image_file(data_file):
    """
    Reads the big endian Mnist file and comverts it into little endian numpy array of images.
    """
    with open(data_file,'rb') as f:
        data = f.read(4)
        magic_no = struct.unpack('>L',data)
        if magic_no==2049 or magic_no==2051:
            print 'Incorrectly parsing files'
        print ' magic no = %d '%magic_no
        data = f.read(4)
        num_data, = struct.unpack('>L',data)
        print ' Number of data points = %d '% num_data
        data = f.read(4)
        rows, = struct.unpack('>L',data)
        data = f.read(4)
        cols, = struct.unpack('>L',data)
        vec_len = rows*cols
        print ' The number of rows = %d and num of cols = %d'% (rows,cols)
        unpacked_data = np.zeros((num_data,vec_len),np.int16)
        for i in range(num_data):
            for j in range(vec_len):
                temp_data, = struct.unpack('>B',f.read(1))
                unpacked_data[i,j] = temp_data

    return unpacked_data

def read_label_file(label_file):
    """
    Reads the big endian Label file and converts it into big endian 
    """
    with open(label_file,'rb') as f:
        data = f.read(4)
        magic_no = struct.unpack('>L',data)
        if magic_no==2049 or magic_no==2051:
            print 'Incorrectly parsing files'
        print ' magic no = %d '%magic_no
        data = f.read(4)
        num_data, = struct.unpack('>L',data)
        print ' Number of data points = %d '% num_data
        unpacked_data = np.zeros((num_data),np.uint8)
        for i in range(num_data):
            temp_data, = struct.unpack('>B',f.read(1))
            unpacked_data[i] = temp_data

    return unpacked_data

#%% 
## To process the training Data
train_data = read_image_file(train_data_file)
train_lab  = read_label_file(train_lab_file)
train_lab.reshape((60000,1))
train_data_comb = np.zeros((60000,785))
train_data_comb[:,:784]= train_data
train_data_comb[:,784]= train_lab.transpose()
from sklearn.cross_validation import train_test_split
train_data,test_data = train_test_split(train_data_comb,test_size=0.16,random_state = 40)
train_lab = train_data[:,784] 
test_lab = test_data[:,784]
train_lab = np.array(train_lab,np.int)
test_lab = np.array(test_lab,np.int)
train_data = train_data[:,:784]
test_data = test_data[:,:784]

#%% 
# Multi Layer Perceptron Code
X = (train_data-128)/255.0
y = train_lab
D = train_data.shape[1]
K = 10
h = 100 # size of hidden layer
W = np.random.uniform(low=-np.sqrt(6./(D+h)),high=np.sqrt(6./(D+h)),size=(D,h)) 
b = np.zeros((1,h))
W2 = np.random.uniform(low=-np.sqrt(6./(K+h)),high=np.sqrt(6./(K+h)),size=(h,K))
b2 = np.zeros((1,K))

# some hyperparameters
step_size = 1e-01
#reg = 1e-3 # regularization strength
reg = 0
# gradient descent loop
num_examples = X.shape[0]
eps = 1e-4
i = 0
loss_prev = np.infty
while(True):
    ## FeedForward Code
    hidden_layer = np.maximum(0, np.dot(X, W) + b) # note, ReLU activation
    scores = np.dot(hidden_layer, W2) + b2
    
    # compute the class probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
    corect_logprobs = -np.log(probs[range(num_examples),y])
    data_loss = np.sum(corect_logprobs)/num_examples
    reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
    loss = data_loss + reg_loss
    if abs(loss_prev-loss)<eps:
        print 'Multi Layer perceptron Converged in %d iterations'%(i)        
        break
        
    loss_prev = loss
    
    if i % 10 == 0:
        print "iteration %d: loss %f" % (i, loss)
    # compute the gradient on scores
    dscores = probs
    dscores[range(num_examples),y] -= 1
    dscores /= num_examples
    
    # backpropate the gradient to the parameters
    # first backprop into parameters W2 and b2
    dW2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)
    # next backprop into hidden layer
    dhidden = np.dot(dscores, W2.T)
    # backprop the ReLU non-linearity
    dhidden[hidden_layer <= 0] = 0
    # finally into W,b
    dW = np.dot(X.T, dhidden)
    db = np.sum(dhidden, axis=0, keepdims=True)
    
    # add regularization gradient contribution
    dW2 += reg * W2
    dW += reg * W
    
    # perform a parameter update
    W += -step_size * dW
    b += -step_size * db
    W2 += -step_size * dW2
    b2 += -step_size * db2
    i+=1
#%%
X_test =(test_data-128)/255.0
y = test_lab
hidden_layer = np.maximum(0, np.dot(X_test, W) + b)
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print 'testing accuracy: %.2f' % (np.mean(predicted_class == y)) 
    

#%%
# Multi Layer Perceptron Code 3 Layer
X = (train_data-128)/255.0
y = train_lab
D = train_data.shape[1]
K = 10
h = 100 # size of hidden layer


W = np.random.uniform(low=-np.sqrt(6./(D+h)),high=np.sqrt(6./(D+h)),size=(D,h)) 
b = np.zeros((1,h))
h = 100 # size of 2nd hidden layer
W1 = np.random.uniform(low=-np.sqrt(6./(h+h)),high=np.sqrt(6./(h+h)),size=(h,h)) 
b1 = np.zeros((1,h))


W2 = np.random.uniform(low=-np.sqrt(6./(K+h)),high=np.sqrt(6./(K+h)),size=(h,K))
b2 = np.zeros((1,K))

#%%


# some hyperparameters
step_size = 1e-01
#reg = 1e-3 # regularization strength
reg = 0
eps = 1e-06
# gradient descent loop
num_examples = X.shape[0]
loss_prev = np.infty
W_v = 0
W1_v = 0
W2_v = 0
b1_v = 0
b2_v = 0
mu = 0.9

i = 0
while(True):
    ## FeedForward Code
    hidden_layer1 = np.maximum(0, np.dot(X, W) + b) # note, ReLU activation
    hidden_layer2 = np.maximum(0, np.dot(hidden_layer1, W1) + b1)     
    scores = np.dot(hidden_layer2, W2) + b2
    
    # compute the class probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
    corect_logprobs = -np.log(probs[range(num_examples),y])
    data_loss = np.sum(corect_logprobs)/num_examples
    reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
    loss = data_loss + reg_loss
    if (abs(loss_prev-loss)<eps):
        print '2 Hidden Layer NN Converged in %d iterations' %i
        break
    if i % 10 == 0:
        print "iteration %d: loss %f" % (i, loss)
        
    loss_prev=loss
    # compute the gradient on scores
    dscores = probs
    dscores[range(num_examples),y] -= 1
    dscores /= num_examples
    
    # backpropate the gradient to the parameters
    # first backprop into parameters W2 and b2
    dW2 = np.dot(hidden_layer2.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)
    # next backprop into hidden layer 2
    dhidden2 = np.dot(dscores, W2.T)
    # backprop the ReLU non-linearity
    dhidden2[hidden_layer2 <= 0] = 0
    
    dW1 = np.dot(hidden_layer1.T,dhidden2)
    db1 = np.sum(dhidden2,axis=0,keepdims=True)    
    # next backprop into hidden layer 1
    dhidden1 = np.dot(dhidden2, W1.T)
    # backprop the ReLU non-linearity
    dhidden1[hidden_layer1 <= 0] = 0
    
    
    # finally into W,b
    dW = np.dot(X.T, dhidden1)
    db = np.sum(dhidden1, axis=0, keepdims=True)
    
    # add regularization gradient contribution
    dW2 += reg * W2
    dW1 += reg*W1
    dW += reg * W
    
    # perform a parameter update
    W += -step_size * dW
    b += -step_size * db
    W1 += -step_size * dW1
    b1 += -step_size * db1
        
    W2 += -step_size * dW2
    b2 += -step_size * db2
    i+=1
#%%
X_test =(test_data-128)/255.0
y = test_lab
hidden_layer1 = np.maximum(0, np.dot(X_test, W) + b)
hidden_layer2 = np.maximum(0, np.dot(hidden_layer1, W1) + b1)
scores = np.dot(hidden_layer2, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print 'testing accuracy: %.2f' % (np.mean(predicted_class == y)) 

#%%
#%%


# some hyperparameters
step_size = 1e-01
reg = 1e-3 # regularization strength

# gradient descent loop
num_examples = X.shape[0]

W_v = 0
W1_v = 0
W2_v = 0
b1_v = 0
b2_v = 0
b_v=0
mu = 0.9

i = 0
while(i<2000):
    ## FeedForward Code
    hidden_layer1 = np.maximum(0, np.dot(X, W) + b) # note, ReLU activation
    hidden_layer2 = np.maximum(0, np.dot(hidden_layer1, W1) + b1)     
    scores = np.dot(hidden_layer2, W2) + b2
    
    # compute the class probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
    corect_logprobs = -np.log(probs[range(num_examples),y])
    data_loss = np.sum(corect_logprobs)/num_examples
    reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
    loss = data_loss + reg_loss
    if i % 10 == 0:
        print "iteration %d: loss %f" % (i, loss)
    
    # compute the gradient on scores
    dscores = probs
    dscores[range(num_examples),y] -= 1
    dscores /= num_examples
    
    # backpropate the gradient to the parameters
    # first backprop into parameters W2 and b2
    dW2 = np.dot(hidden_layer2.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)
    # next backprop into hidden layer 2
    dhidden2 = np.dot(dscores, W2.T)
    # backprop the ReLU non-linearity
    dhidden2[hidden_layer2 <= 0] = 0
    
    dW1 = np.dot(hidden_layer1.T,dhidden2)
    db1 = np.sum(dhidden2,axis=0,keepdims=True)    
    # next backprop into hidden layer 1
    dhidden1 = np.dot(dhidden2, W1.T)
    # backprop the ReLU non-linearity
    dhidden1[hidden_layer1 <= 0] = 0
    
    
    # finally into W,b
    dW = np.dot(X.T, dhidden1)
    db = np.sum(dhidden1, axis=0, keepdims=True)
    
    # add regularization gradient contribution
    dW2 += reg * W2
    dW1 += reg*W1
    dW += reg * W
    
    # perform a parameter update
    
    W1_v= W1_v*mu -step_size*dW1   
    W1+= W1_v

    W_v= W_v*mu -step_size*dW   
    W+= W_v
    
    W2_v= W2_v*mu -step_size*dW2   
    W2+= W2_v
    
    b_v = b_v*mu -step_size*db
    b+=b_v
    
    b1_v = b1_v*mu -step_size*db1
    b1+=b1_v
    
    b2_v = b2_v*mu -step_size*db2
    b2+=b2_v
    
    
    #W += -step_size * dW
    #b += -step_size * db
    #W1 += -step_size * dW1
    #b1 += -step_size * db1
        
    #W2 += -step_size * dW2
    #b2 += -step_size * db2
    i+=1





