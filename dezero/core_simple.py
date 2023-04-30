import weakref
import numpy as np
import contextlib


# =============================================================================
# Config
# =============================================================================
class Config:
    enable_backprop = True # 역전파 가능 여부, 'True' : 역전파 활성화 모드

@contextlib.contextmanager
def using_config(name, value): # name : 사용할 Config 속성의 이름
    old_value = getattr(Config, name) # name을 getattr 함수에 넘겨 Config 클래스에서 꺼내온다.
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad(): # 편의 함수
    return using_config('enable_backprop', False)


# =============================================================================
# Variable / Function
# =============================================================================
class Variable:
    __array_priority__ = 200 # 연산자 우선순위 지정

    def __init__(self, data, name = None): # 인스턴스 변수
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))
        self.data = data
        self.name = name
        self.grad = None # 미분값 저장
        self.creator = None # 인스턴스 변수 추가
        self.generation = 0 # 세대 수를 기록하는 변수

    @property # shape 메서드를 인스턴스 변수처럼 사용하게한다.
    def shape(self):
        return self.data.shape # 인스턴스 변수로 데이터의 형상을 얻을 수 있다.

    @property 
    def ndim(self): # 차원 수
        return self.data.ndim 

    @property 
    def size(self): # 원소 수
        return self.data.size

    @property  
    def dtype(self): # 데이터 타입
        return self.data.dtype

    def __len__(self): # 객체 수 알림
        return len(self.data) 

    def __repr__(self): # ndarray에 담긴 내용 출력
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9) # 줄을 바꾸고 새로운 줄 앞에 공백 9개를 삽입
        return 'variable(' + p + ')'

    def set_creator(self, func): # 메서드 추가
        self.creator = func
        self.generation = func.generation + 1 # 세대를 기록 (부모 세대 + 1)

    def cleargrad(self):
        self.grad = None # 여러 가지 미분을 계산할 때 똑같은 변수를 재사용 가능

    def backward(self, retain_grad = False): # retain_grad = False는 중간 변수 미분값을 모두 None으로 설정
        if self.grad is None:
            self.grad = np.ones_like(self.data) # self.data와 형상과
            # 데이터 터입이 같은 ndarray 인스턴스 생성

        funcs = []
        seen_set = set() # 함수의 backward 메서드가 여러 번 불리는 것 방지

        def add_func(f): # 함수 리스트 세대 순으로 정렬
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)
        
        while funcs:
            f = funcs.pop() # 1. 함수를 가져온다.
            gys = [output().grad for output in f.outputs] # 약한 참조를 적용하고 출력 변수(미분값)을 리스트에 담는다.
            gxs = f.backward(*gys) # 역전파 호출 : 리스트 언팩
            if not isinstance(gxs, tuple): # 튜플이 아닌 경우 튜플로 변경
                gxs = (gxs, )
            for x, gx in zip(f.inputs, gxs): # 모든 Variable 인스턴스 각각에 알맞은 미분값을 설정
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx # 미분값을 더해준다.

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None # y는 약한 참조, 각 함수의 출력 변수의 미분값을 
                    # 유지하지 않도록 설정

# 편의 함수
# def as_variable(obj):
#     if isinstance(obj, Variable): # obj가 Variable 인스턴스 인지 확인
#         return obj
#     return Variable(obj) # 만약 Variable 인스턴스가 아니라면 변환

def as_variable(obj): 
    if isinstance(obj, Variable): # obj가 Variable 인스턴스 인지 확인
        return obj
    return Variable(obj) # 만약 Variable 인스턴스가 아니라면 변환

# 편의 함수
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, *inputs): # 임의 개수의 인수(가변 길이 인수)를 건네 함수를 호출할 수 있다.
        inputs = [as_variable(x) for x in inputs] # inputs에 담긴 원소를 Variable로 변환
        xs = [x.data for x in inputs] # input을 리스트로
        ys = self.forward(*xs) # 언팩
        if not isinstance(ys, tuple): # 튜플이 아닌 경우 튜플로 변경
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys] # output을 리스트로

        if Config.enable_backprop:

            # 입력 변수와 같은 값으로 generation 설정
            # '역전파 비활성화 시' 세대 설정은 필요하지 않다. 
            self.generation = max([x.generation for x in inputs]) # 세대 설정

            for output in outputs:
                output.set_creator(self) # 연결 설정
            self.inputs = inputs # 순전파 때 결과값 기억
            self.outputs = [weakref.ref(output) for output in outputs] # 약한 참조

            # 리스트의 원소가 하나라면 첫 번째 원소를 반환한다.
            return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError() 

    def backward(self, gy):
        raise NotImplementedError()


# =============================================================================
# 사칙연산 / 연산자 오버로드
# =============================================================================
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    x1 = as_array(x1) # ndarray 인스턴스로 변환
    return Add()(x0, x1)

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x0, gy * x1

def mul(x0, x1):
    x1 = as_array(x1) # ndarray 인스턴스로 변환
    return Mul()(x0, x1)

class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy

def neg(x):
    return Neg()(x)

class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y

    def backward(self, gy):
        return gy, -gy

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0) # x0와 x1의 순서를 바꿔준다.

class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        return gx0, gx1

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0) # x0과 x1의 순서를 바꾼다.

class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        c = self.c
        gx = c * x ** (c-1) * gy
        return gx

def pow(x, c):
    return Pow(c)(x)

def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
