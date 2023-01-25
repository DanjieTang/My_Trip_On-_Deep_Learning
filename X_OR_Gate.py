import numpy as np

def first_perceptron(x):
  w=np.array([-1, 1])
  bias=1
  return np.dot(x, w)+bias

def second_perceptron(x):
  w=np.array([1, -1])
  bias=1
  return np.dot(x, w)+bias

def andGate(x1, x2):
  if(x1 and x2):
    return 1
  else:
    return 0

def XOR(x):
  x1=first_perceptron(x)
  x2=second_perceptron(x)
  return andGate(x1, x2)

point1=np.array([0, 0])
point2=np.array([0, 1])
point3=np.array([1, 0])
point4=np.array([1, 1])

print(XOR(point1))
print(XOR(point2))
print(XOR(point3))
print(XOR(point4))