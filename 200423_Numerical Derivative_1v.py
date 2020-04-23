
# 예제 1. 1변수 함수 f(x) = x^2, f'(3.0)

import numpy as np

def numerical_derivative(f, x) :
    delta_x = 1e-4
    grad = np.zeros_like(x)
    print ("debug1. initial input variable= ", x)
    print ("debug2. initial grad= ", grad)
    print ("===================================")

    it = np.nditer(x, flags=["multi_index"], op_flags=['readwrite'])

    while not it.finished :
        idx = it.multi_index

        print("debug3. idx= ", idx, ", x[idx] = ", x[idx])

        tmp_val = x[idx] # 초기값 백업
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x)

        x[idx] = tmp_val - delta_x
        fx2 = f(x)
        grad[idx] = (fx1 - fx2) / (2*delta_x)

        print ("debug4. grad[idx] = ", grad[idx])
        print ("debug5. grad = ", grad)
        print ("===================================")
        
        x[idx] = tmp_val # 변한 x[idx]값을 원래값으로 되돌려줌.
        it.iternext()

    return grad

def func1(input_obj) : # input_obj : 벡터나 행렬을 나타내는 numpy 객체
    x = input_obj[0]
    return x ** 2

result = numerical_derivative(func1, np.array([3.0]))
print (result)