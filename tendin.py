from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import svm,metrics
from sklearn import linear_model
np.random.seed(2)

X = np.array([[2.3137,2.1933,1.2287,1.8942,1.0884,0.8857,3.7036,3.202,0.5966,1.4794,
               0.4807,0.3113,0.5194,0.3196,0.5685,0.7499,0.5406,1.3403,0.2914,0.583]])
y = np.array([1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0])
#test
X_test =np.array([[0.6214,1.6378,3.5238,1.0753,1.326,0.5925,0.6415,0.9196,0.8348,1.7442]]) 
y_test =np.array([1,1,1,1,1,0,0,0,0,0])
# extened data 
X = np.concatenate((np.ones((1, X.shape[1])), X), axis = 0)
X_test = np.concatenate((np.ones((1, X_test.shape[1])), X_test), axis = 0)
def sigmoid(s):
    return 1/(1 + np.exp(-s))

def logistic_sigmoid_regression(X, y, w_init, eta, tol = 1e-4, max_count = 10000):
    w = [w_init]    
    it = 0
    N = X.shape[1]
    d = X.shape[0]
    count = 0
    check_w_after = 20
    while count < max_count:
#         it += 1
        # mix data 
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = y[i]
            zi = sigmoid(np.dot(w[-1].T, xi))
            w_new = w[-1] + eta*(yi - zi)*xi
            count += 1
            # stopping criteria
            if count%check_w_after == 0:                
                if np.linalg.norm(w_new - w[-check_w_after]) < tol:
                    return w
            w.append(w_new)
    return w
eta = .05 
d = X.shape[0]
w_init = np.random.randn(d, 1)

w = logistic_sigmoid_regression(X, y, w_init, eta)
print(w[-1])

print(sigmoid(np.dot(w[-1].T, X_test)))

#predict
m=sigmoid(np.dot(w[-1].T, X_test))
print("PREDICT")

if m.any() <= 0.5:
    predict=0
else:
    predict =1
print("RESULT")
def acc(y_test, predict):
    correct = np.sum(y_test == predict)
    return float(correct)/y_test.shape[0]
print('accuracy = ', acc(y_test, predict))