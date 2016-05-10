import numpy as np
from keras.datasets import mnist

# initialize parameters randomly
D = 784
h = 100 # size of hidden layer
W = np.random.uniform(low=-np.sqrt(6.0/(D+h)), high=np.sqrt(6.0/(D+h)), size=(D, h))
b = np.zeros((1,h))
k=10;

b1 = np.zeros((1,h))
W1 = np.random.uniform(low=-np.sqrt(6.0/(h+h)), high=np.sqrt(6.0/(h+h)), size=(h, h))
W2 = np.random.uniform(low=-np.sqrt(6.0/(k+h)), high=np.sqrt(6.0/(k+h)), size=(h, k))
b2 = np.zeros((1,k))

# some hyperparameters
step_size = 1e-1
reg = 1e-3 # regularization strength

###################
#READ MNIST data
(X, y), (test_feat, test_label) = mnist.load_data()

X = np.array(X, np.float)
X = X.reshape(X.shape[0], 784)

test_feat = np.array(test_feat, np.float)
test_feat = test_feat.reshape(test_feat.shape[0], 784)

X = (X - 128)/255.0
test_feat = (test_feat-128)/255.0
###################

vW = 0
vB = 0
vW2 = 0
vB2 = 0
# gradient descent loop
num_examples = X.shape[0]

W_v = 0
W1_v = 0
W2_v = 0
b1_v = 0
b2_v = 0
mu = 0.9

i = 0

loss_prev = 10

for i in range(600):
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

    loss_prev = loss	
    if i % 10 == 0:
        print ("iteration {}: loss {}".format(i, loss))
        
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
X_test =(test_feat-128)/255.0

y_2 = test_label
y_1 = y

hidden_layer1 = np.maximum(0, np.dot(X_test, W) + b)
hidden_layer2 = np.maximum(0, np.dot(hidden_layer1, W1) + b1)

scores = np.dot(hidden_layer2, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print ('testing accuracy: {}'.format(np.mean(predicted_class == y_2)) )


hidden_layer1 = np.maximum(0, np.dot(X, W) + b)
hidden_layer2 = np.maximum(0, np.dot(hidden_layer1, W1) + b1)

scores = np.dot(hidden_layer2, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print ('train accuracy: {}'.format(np.mean(predicted_class == y_1)) )

