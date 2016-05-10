import numpy as np

def dot_product(a, b):
    return np.array(np.dot(np.asarray(a), np.asarray(b)))	


def decision( x, w, theta ):
    return (dot_product(x, w) > theta)


def perceptron( training_data ):
    theta = 1
    iteration = 0
    weights = [0.01, 0.01, 0.01, 0.01]
    converged = False
    
    while not converged:
	correct_count = 0;
        for key, val in training_data.iteritems():
            d = decision(key, weights, theta)
            if d == val:
		correct_count +=1 
                continue
            elif d == False and val == True:
                theta -= 1
		iteration += 1
                for i in range(len(key)):
                    weights[i] += key[i]

            elif d == True  and val == False:
                theta += 1
                iteration += 1
                for i in range(len(key)):
                    weights[i] -= key[i]

        if (correct_count == len(training_data)):
	    break;

    print ("Converged in Iterations {}".format(iteration))	
    return weights, theta

lines = [line.rstrip('\n') for line in open('iris/iris_train.data')]

train = {}
for line in lines:
    words = line.split(',')
    tup   = (float(words[0]),float(words[1]),float(words[2]),float(words[3]),) 
    if (words[4] == 'Iris-setosa'):
	train[tup] = True
    else: 
        train[tup] = False

lines1 = [line.rstrip('\n') for line in open('iris/iris_test.data')]

test = {}
for line in lines1:
    words = line.split(',')
    tup   = (float(words[0]),float(words[1]),float(words[2]),float(words[3]),)
    if (words[4] == 'Iris-setosa'):
        test[tup] = True
    else:
        test[tup] = False

#print (train)
#print (test)


train_feat = np.asarray(train.keys())
test_feat = np.asarray(test.keys())

train_mean = np.mean(train_feat, axis=0)
test_mean = np.mean(test_feat, axis=0)
train_std = np.std(train_feat, axis=0)
test_std = np.std(test_feat, axis=0)

train_z = {}
for key, val in train.iteritems():
	key_new = tuple(np.array((key-train_mean)/train_std))
	train_z[key_new] = val

test_z = {}
for key, val in test.iteritems():
        key_new = tuple(np.array((key-test_mean)/test_std))
        test_z[key_new] = val
	


weights, theta = perceptron( train )
total_correct = 0
for key, val in test.iteritems():
    d = decision(key, weights, theta)
    if d == val:
	total_correct += 1
    
print ("No Z scoring\n")
print ("Total Correct = {}, out of {}".format(total_correct, len(test))) 

print ("\n\n")


weights1, theta1 = perceptron( train_z )
total_correct = 0
for key, val in test_z.iteritems():
    d = decision(key, weights1, theta1)
    if d == val:
        total_correct += 1
   
print ("With Z scoring\n")
print ("Total Correct = {}, out of {}".format(total_correct, len(test_z)))
