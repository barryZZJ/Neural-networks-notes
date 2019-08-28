from math import exp

def sigmoid(z):
    return round(1/(1+exp(-z)),4)
def step_function(z):
    return z
def product(W,X):#矩阵乘法
    res=0
    for i in range(len(W)):
        res+=W[i]*X[i]
    return res
def forward(X):
    dea=0
    ACT=10
    threshold=-5
    w1=[dea,dea,dea,dea,dea,dea,dea,dea,ACT,ACT]
    w2=[dea,dea,dea,dea,ACT,ACT,ACT,ACT,dea,dea]
    w3=[dea,dea,ACT,ACT,dea,dea,ACT,ACT,dea,dea]
    w4=[dea,ACT,dea,ACT,dea,ACT,dea,ACT,dea,ACT]
    b1=threshold
    b2=threshold
    b3=threshold
    b4=threshold
    W=[w1,w2,w3,w4]
    B=[b1,b2,b3,b4]
    O=[]
    for j in range(4):
        #第j个神经元
        tmp = product(W[j],X)+B[j]
        o=sigmoid(product(W[j],X)+B[j]) #output
        #o=step_function(product(W[j],X)+B[j]) #output
        O.append(o)
    return O
def neu2bin(O):
    res = ""
    for i in O:
        if i>=0.5:
            res+='1'
        else:
            res+='0'
    return res
def getX(num):
    res = []
    for i in range(l):
        if i==num:
            res.append(0.99)
        else:
            res.append(0.01)
    return res
l=10
print(neu2bin(forward(getX(1))))
exit()