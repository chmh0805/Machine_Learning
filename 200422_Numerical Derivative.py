
'''
#수치미분은 수학공식을 쓰지 않고 C 또는 Python을 이용하여, 주어진 입력 값이
미세하게 변할 때 함수값 f는 얼마나 변하는 지를 계산하게 해주는 것을 지칭


# 수치미분 구현 1차버전
def numerical_derivative(f, x) :
# f는 미분하려는 함수, 외부에서 def,lambda 등으로 정의됨
    delta_x = 1e-4 # lim에 해당되는 작은 값
    return (f(x+delta_x) - f(x-delta_x)) / (2*delta_x)

'''

# 예제 1
# 함수 f(x)=x^2에서 미분계수 f'(3)을 구하기.
## 즉, x=3에서 값이 미세하게 변할 때, 함수 f는 얼마나 변하는지 계산하라는 의미

def my_func1(x) :
    return x**2 # f(x) = x^2

def numerical_derivative(f, x) :
    delta_x = 1e-4
    return (f(x+delta_x) - f(x-delta_x)) / (2 * delta_x)

result = numerical_derivative(my_func1, 3)
print ("result == ", result)