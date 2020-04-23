
import matplotlib.pyplot as plt # matplotlib에서 pyplot 오픈 (plt)
import numpy as np # numpy 오픈 (np)


for i in range (100,111) :
    x_data = np.random.rand(i)
    y_data = np.random.rand(i)
    plt.title = 'scatter {}'.format(i)
    plt.grid()
    plt.scatter(x_data, y_data, color='b')
    plt.show()

"""
x_data = np.random.rand(150) # x값(변수) 지정
y_data = np.random.rand(150) # y값(변수) 지정

plt.title = 'scatter 1' # 그래프 제목 생성
plt.grid() # 격자눈금 생성
plt.scatter(x_data,y_data,color='b') # scatter(점형) 그래프 생성 (x값, y값, 표시 색깔)
plt.show() # 그래프 보여주기
"""