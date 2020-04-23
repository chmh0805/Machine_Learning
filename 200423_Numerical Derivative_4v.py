
# 예제 3. 4변수함수 f(w,x,y,z) = wx + xyz + 3w + zy^2를 미분하여
## f'(1.0, 2.0, 3.0, 4.0)의 값을 구하여라.

import numpy as np

def numerical_derivative(f, x) :
    delta_x = 1e-4
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"],op_flags=["readwrite"])

    while not it.finished :
        idx = it.multi_index

        tmp_val = x[idx]
        x[idx] = tmp_val + delta_x
        fx1 = f(x)

        x[idx] = tmp_val - delta_x
        fx2 = f(x)

        grad[idx] = (fx1 - fx2) / (2 * delta_x)

        x[idx] = tmp_val
        it.iternext()

    return grad

def func3(input_obj) :
    w = input_obj[0,0]
    x = input_obj[0,1]
    y = input_obj[1,0]
    z = input_obj[1,1]
    return ( w*x + x*y*z + 3*w + z*np.power(y,2) )

input = np.array( [[1.0, 2.0], [3.0, 4.0]])

result = numerical_derivative(func3, input)
print (result)