
import numpy as np

# External Function

def sigmoid (x) :
    return 1 / (1 + np.exp(-x))

def numerical_derivative(f,x) :
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

# LogicGate Class

class LogicGate :
    def __init__(self, gate_name, xdata, tdata) :
        self.name = gate_name
        # 입력데이터, 정답데이터 초기화
        self.__xdata = xdata.reshape(4,2)
        self.__tdata = tdata.reshape(4,1)
        # 2층 hidden layer unit(은닉층) : 6개 가정, W2,b2 초기화
        self.__W2 = np.random.rand(2,6) # weight : 2 * 6 matrix
        self.__b2 = np.random.rand(6)
        # 3층 output layer unit(출력층) : W3,b3 초기화
        self.__W3 = np.random.rand(6,1)
        self.__b3 = np.random.rand(1)
        # 학습률 learning_rate 초기화
        self.__learning_rate = 1e-2

    def feed_forward(self) : # feed forward를 통하여 손실함수(cross-entropy)값 계산
        delta = 1e-7 # log 무한대 발산 방지
        
        z2 = np.dot(self.__xdata, self.__W2) + self.__b2 # 은닉층의 선형회귀 값
        a2 = sigmoid(z2) # 은닉층의 출력

        z3 = np.dot(a2, self.__W3) + self.__b3 # 출력층의 선형회귀 값
        y = a3 = sigmoid(z3) # 출력층의 출력
        # cross-entropy
        return -np.sum ( self.__tdata * np.log(y+delta) + (1-self.__tdata)*np.log((1-y)+delta) )

    def loss_val(self) :
        delta = 1e-7
        
        z2 = np.dot(self.__xdata, self.__W2) + self.__b2
        a2 = sigmoid(z2)

        z3 = np.dot(a2, self.__W3) + self.__b3
        y = a3 = sigmoid(z3)
        # cross-entropy
        return -np.sum ( self.__tdata * np.log(y+delta) + (1 - self.__tdata)*np.log((1-y)+delta) )


    def train(self) :
        f = lambda x : self.feed_forward()
        for step in range (10001) :
            self.__W2 -= self.__learning_rate * numerical_derivative(f, self.__W2)
            
            self.__b2 -= self.__learning_rate * numerical_derivative(f, self.__b2)

            self.__W3 -= self.__learning_rate * numerical_derivative(f, self.__W3)
            
            self.__b3 -= self.__learning_rate * numerical_derivative(f, self.__b3)
            
            if (step % 400 == 0) :
                print ("step = ",step, ", loss value = ", self.loss_val())

    def predict(self, xdata) :

        z2 = np.dot(xdata, self.__W2) + self.__b2
        a2 = sigmoid(z2)

        z3 = np.dot(a2, self.__W3) + self.__b3
        y = a3 = sigmoid(z3)

        if y > 0.5 :
            result = 1
        else :
            result = 0

        return y, result

# Usage
# <AND GATE>
xdata = np.array([ [0,0], [0,1], [1,0], [1,1] ])
tdata = np.array([0, 0, 0, 1])
AND_obj = LogicGate("AND_GATE", xdata, tdata)
AND_obj.train()

print (AND_obj.name,"\n")
test_data = np.array([ [0,0], [0,1], [1,0], [1,1] ])
for data in test_data :
    print (AND_obj.predict(data))
