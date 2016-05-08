import numpy as np
from keras.datasets import mnist

# initialize parameters randomly
D = 784
h = 100 # size of hidden layer
W = np.random.uniform(low=-np.sqrt(6.0/(D+h)), high=np.sqrt(6.0/(D+h)), size=(D, h))
b = np.zeros((1,h))
k=10;
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

prev_loss = 0.0;
enable_regularization = 1
enable_momentum = 1

#%%
for i in range(10000):
  
  # evaluate class scores, [N x K]
  hidden_layer = np.maximum(0, np.dot(X, W) + b) # note, ReLU activation
  scores = np.dot(hidden_layer, W2) + b2
  
  # compute the class probabilities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
  
  # compute the loss: average cross-entropy loss and regularization
  corect_logprobs = -np.log(probs[range(num_examples),y])
  data_loss = np.sum(corect_logprobs)/num_examples
  reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
  loss = data_loss + reg_loss
  if i % 10 == 0:
      print('iteration {} :: loss {}'.format(i, loss))  
      
  if (loss - prev_loss < 0.001):
      print('iteration {} :: loss {}'.format(i, loss))  
      prev_loss = loss
      break
  
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
  
#  # add regularization gradient contribution
  if enable_regularization == 1:
      dW2 += reg * W2
      dW += reg * W
  
  
###Add momentum  
  vW = 0.9 * vW - step_size * dW 
  vW2 = 0.9 * vW2 - step_size * dW2 
  vB = 0.9 * vB - step_size * db 
  vB2 = 0.9 * vB2 - step_size * db2 
  
  # perform a parameter update
  if enable_momentum == 1:
      W += vW
      b += vB
      W2 += vW2
      b2 += vB2
  else:     
      W += -step_size * dW
      b += -step_size * db
      W2 += -step_size * dW2
      b2 += -step_size * db2
  
##############
#%%
hidden_layer = np.maximum(0, np.dot(test_feat, W) + b)
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print ('test accuracy: {}'.format(np.mean(predicted_class == test_label)))

hidden_layer = np.maximum(0, np.dot(X, W) + b)
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print ('train accuracy: {}'.format(np.mean(predicted_class == y)))
