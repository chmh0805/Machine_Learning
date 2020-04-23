
# 예제 2. 2변수 함수 f(x,y) = 2x + 3xy + y^3, f'(1.0,2.0)

import numpy as np

def numerical_derivative(f, x):
    delta_x = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])

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

def func2(input_obj) :
    x = input_obj[0]
    y = input_obj[1]
    return (2*x + 3*x*y + np.power(y,3))

input = np.array([1.0, 2.0])
result = numerical_derivative(func2, input)
print (result)