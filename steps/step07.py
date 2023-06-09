# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dWbkE1ucKOUJl3uwzb_-bBs5lcwgz_L2
"""
import numpy as np

class Variable():
    def __init__(self, data):
        self.data = data
        self.grad = None # 미분값 저장
        self.creator = None # 인스턴스 변수 추가

    def set_creator(self, func): # 메서드 추가
        self.creator = func

    def backward(self):
        f = self.creator # 1. 함수를 가져온다.
        if f is not None:
            x = f.input # 2. 함수의 입력을 가져온다.
            x.grad = f.backward(self.grad) # 3. 함수의 backward 메서드를 호출한다.
            x.backward() # 하나 앞 변수의 backward 메서드를 호출한다. (재귀)

class Function:
    def __call__(self, input): # __call__ 메서드의 인수 input은 Variable 인스턴스라고 가정
        x = input.data # 데이터를 꺼낸다
        y = self.forward(x) # 구체적인 계산은 forward 메서드에서 한다.
        output = Variable(y) # Variable 형태로 되돌림
        output.set_creator(self) # 출력 변수에 창조자를 설정
        self.input = input # 입력 변수를 기억한다.
        self.output = output # 출력 변수를 기억한다.
        return output

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy # y = x^2의 미분은 2 * x
        return gx

class Exp(Function):
    def forward(self, x):
        y = np.exp(x) # 계산 값
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# 계산 그래프의 노드들을 거꾸로 거슬러 올란다.
# assert y.creator == C
# assert y.creator.input == b
# assert y.creator.input.creator == B
# assert y.creator.input.creator.input == a
# assert y.creator.input.creator.input.creator == A
# assert y.creator.input.creator.input.creator.input == x

y.grad = np.array(1.0)

# C = y.creator # 1. 함수를 가져온다
# b = C.input # 2. 함수의 입력을 가져온다.
# b.grad = C.backward(y.grad) # 3. 함수의 backward 메서드를 호출한다.

# B = b.creator # 1. 함수를 가져온다
# a = B.input # 2. 함수의 입력을 가져온다
# a.grad = B.backward(b.grad)

# A = a.creator # 1. 함수를 가져온다
# x = A.input # 2. 함수의 입력을 가져온다
# x.grad = A.backward(b.grad)
y.backward()
print(x.grad)