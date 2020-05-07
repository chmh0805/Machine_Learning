
import numpy as np

x_data = np.array([ [2,4],[4,11],[6,6],[8,5],[10,7],[12,16],[14,8],[16,3],[18,7] ])
t_data = np.array([0,0,0,0,1,1,1,1,1]).reshape(9,1)

W = np.random.rand(2,1)
b = np.random.rand(1)

def sigmoid(x) :
    return 1 / (1 + np.exp(-x))

def loss_func(x, t) :
    delta = 1e-7
    z = np.dot(x, W) + b
    y = sigmoid(z)
    return -np.sum( t*np.log(y+delta) + ((1-t) * np.log(1-y+delta)) )

def numerical_derivative(f, x) :
    delta_x = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished :
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x)

        x[idx] = tmp_val - delta_x
        fx2 = f(x)
        grad[idx] = (fx1 - fx2) / (2 * delta_x)
        x[idx] = tmp_val
        it.iternext()
    return grad

def error_val(x, t) :
    delta = 1e-7
    z = np.dot(x, W) + b
    y = sigmoid(z)
    return -np.sum( t*np.log(y+delta) + ((1-t) * np.log(1-y+delta)) )

def predict(x) :
    z = np.dot(x, W) + b
    y = sigmoid(z)
    if y > 0.5 :
        result = 1
    else :
        result = 0
    return result

learning_rate = 1e-2
f = lambda x : loss_func(x_data, t_data)

for step in range (60001) :
    W -= learning_rate * numerical_derivative(f, W)
    b -= learning_rate * numerical_derivative(f, b)
    if (step % 400 == 0) :
        print ("step = ",step, "error_value = ",error_val(x_data,t_data), "W = ",W,"b = ",b)


test1 = np.array([3,17]);
test2 = np.array([5, 8]);
test3 = np.array([7,21]);
test4 = np.array([12,0]);
print (predict(test1));
print (predict(test2));
print (predict(test3));
print (predict(test4));