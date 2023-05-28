# Add import path for the dezero directory.

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable

x = Variable(np.array(2.0))
y = x ** 2
y.backward(create_graph = True) # 미분을 하기 위해 역전파하는 코드
gx = x.grad # 계산 그래프 식
x.cleargrad()

z = gx ** 3 + y
z.backward()
print(x.grad)