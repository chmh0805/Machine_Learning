
import numpy as np

def sigmoid(x) :
    return 1 / (1 + np.exp(-x))

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

class LogicGate :
    def __init__(self, gate_name, xdata, tdata) :
        self.name = gate_name

        # 입력데이터, 정답데이터 초기화
        self.__xdata = xdata.reshape(4,2)
        self.__tdata = tdata.reshape(4,1)

        # 가중치 W, 바이어스 b 초기화
        self.__W = np.random.rand(2,1)
        self.__b = np.random.rand(1)

        # 학습률 learning rate 초기화
        self.__learning_rate = 1e-2
    # 손실함수
    def __loss_func(self) :
        delta = 1e-7
        z = np.dot(self.__xdata, self.__W) + self.__b
        y = sigmoid(z)
        return -np.sum( (self.__tdata*np.log(y+delta)) + (1-self.__tdata)*np.log((1-y)+delta) )
    # 손실 값 계산
    def error_val(self) :
        delta = 1e-7
        z = np.dot(self.__xdata, self.__W) + self.__b
        y = sigmoid(z)
        return -np.sum( (self.__tdata*np.log(y+delta)) + (1-self.__tdata) * np.log((1-y)+delta) )
    # 수치미분
    def train(self) :
        f = lambda x : self.__loss_func()
        print ("Initial error value = ",self.error_val())
        for step in range (8001) :
            self.__W -= self.__learning_rate * numerical_derivative(f, self.__W)
            self.__b -= self.__learning_rate * numerical_derivative(f, self.__b)
            if (step % 400 == 0) :
                print("step = ",step, "error value = ",self.error_val())
    # 예측함수
    def predict(self, xdata) :
        z = np.dot(input_data, self.__W) + self.__b
        y = sigmoid(z)
        if y > 0.5 :
            result = 1
        else :
            result = 0
        return y, result


x_data = np.array([ [0,0], [0,1], [1,0], [1,1] ])
t_data = np.array([0,0,0,1])

AND_obj = LogicGate("AND_GATE",x_data,t_data)
AND_obj.train()

print(AND_obj.name, "\n")

test_data = np.array([ [0,0], [0,1], [1,0], [1,1] ])

for input_data in test_data :
    (sigmoid_val, logical_val) = AND_obj.predict(input_data)
    print (input_data, "=", logical_val, "\n")