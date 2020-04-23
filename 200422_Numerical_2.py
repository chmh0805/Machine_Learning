
# 예제2 : 함수 f(x) = 3xe^x를 미분한 함수를 f'(x)라고 할 경우,
## f'(2)를 구하기. x=2에서 값이 미세하게 변할 때, 함수 f는 얼마나 변하는지 계산.

import numpy as np

def my_func2(x) :
    return 3*x*(np.exp(x))

def numerical_derivative(f, x) :
    delta_x = 1e-4
    return (f(x+delta_x) - f(x-delta_x)) / ( 2 * delta_x )

result = numerical_derivative (my_func2, 2)
print ("result == ", result)