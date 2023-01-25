import numpy as np
import math
import random

slope=4
vertical_shift=20

def sigmoid_function(sum):
    return 1/(1+math.e**(-sum))

def simple_ANN(x, w, t, lr, iter):
    total=0
    counter=0
    for i in range(iter):
        lr=(1-counter/iter)*lr+0.03*(counter/iter)

        y=np.array([])
        error=np.array([])
        for i in range(len(x)):
            sum=0
            for j in range(len(x[i])):
                sum+=w[j]*x[i][j]

            sum+=vertical_shift

            y=np.append(y, sigmoid_function(sum))

            error=np.append(error, (y[i]-t[i])**2)

            for j in range(len(x[i])):
                d=x[i][j]*2*(y[i]-t[i])*(y[i]*(1-y[i]))
                w[j]-=d*lr

        counter+=1


    total_e=np.mean(error)
    return (y, w, total_e)

if __name__ == "__main__":
    x=np.array([[-100, -10]])
    t=np.array([0])
    for i in range(500):
        x_part=random.randint(-100, 100)
        y_part=random.randint(-100, 100)
        x=np.append(x, [[x_part, y_part]], axis=0)
        if(y_part>(slope*x_part+vertical_shift)):
            t=np.append(t, 1)
        else:
            t=np.append(t, 0)


    w=np.array([0.3, 0.3])

    answer=simple_ANN(x, w, t, 0.3, 5)
    success_num=0

    for i in range(1000):
        x_part=random.randint(-1000, 1000)
        y_part=random.randint(-1000, 1000)

        sum=x_part*answer[1][0]+y_part*answer[1][1]

        output=sigmoid_function(sum)

        if(output<0.5 and y_part<=(slope*x_part+vertical_shift)):
            success_num+=1
            continue
        elif(output>0.5 and y_part>(slope*x_part+vertical_shift)):
            success_num+=1


    print("The success rate is "+str(success_num/1000))
