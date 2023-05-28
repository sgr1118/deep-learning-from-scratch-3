# Add import path for the dezero directory.

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F

x = Variable(np.array(1.0))
y = F.tanh(x)
x.name = 'x'
y.name = 'y'
y.backward(create_graph = True)

iters = 4 # n차 미분

for i in range(iters):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph = True)

# 계산 그래프 그리기
gx = x.grad
gx.name = 'gs' + str(iters + 1)
plot_dot_graph(gx, verbose = False, to_file = 'tanh.png')