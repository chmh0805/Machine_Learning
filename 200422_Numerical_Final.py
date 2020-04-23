
'''
# 수치미분 최종 버전 - numerical derivative
입력변수가 하나 이상인 다변수 함수의 경우, 입력변수는 서로 독립적이기 떄문에
수치미분 또한 변수의 개수만큼 개별적으로 계산하여야 함.

[예] f(x,y) = 2x + 3xy + y^3이라면, 입력변수 x,y 두 개이므로 각각 수치미분 수행
f'(1.0,2.0)값을 계산하기 위해서는,
=> x = 1.0 에서의 미분계수는 변수 y = 2.0을 상수로 대입하여 수행
=> y = 2.0 에서의 미분계수 또한 변수 x = 1.0을 상수로 대입하여 수행
'''


import numpy as np

def numerical_derivative(f, x) :
    # f는 다변수함수, x는 모든 변수를 포함하고 있는 numpy 객체 (배열, 행렬, ...)
    delta_x = 1e-4
    grad = np.zeros_like(x) # 계산된 수치미분 값을 저장할 변수(그릇), 0으로 초기화
        
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite']) #iterator

    while not it.finished : # 변수의 개수만큼 반복
        idx = it.multi_index

        tmp_val = x[idx] # numpy 타입은 mutable 이므로 원본값 보관
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x) # fx1에 f(x+delta_x) 값 저장

        x[idx] = tmp_val - delta_x
        fx2 = f(x) # fx2에 f(x-delta_x) 값 저장
        grad[idx] = (fx1 - fx2) / (2*delta_x)

        x[idx] = tmp_val
        it.iternext()

    return grad