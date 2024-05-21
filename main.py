import matplotlib.pyplot as plt
import numpy as np
import csv
import math


def linearDistribution(n, distribution):
    array_x = []
    array_y = []
    for i in range(n - 1):
        array_x.append(x_values[i * distribution])
        array_y.append(y_values[i * distribution])
    array_x.append(x_values[-1])
    array_y.append(y_values[-1])
    return [array_x, array_y]


def baseLagrange(x_array, index, x):
    counter = 1
    denominator = 1
    for i in range(len(x_array)):
        if i != index:
            counter *= x - x_array[i]
            denominator *= x_array[index] - x_array[i]

    return counter / denominator


def interpolatedFunc(x_orginal, x_array, y_array, nodes):
    result = []
    for x in range(len(x_orginal)):
        sum_ = 0
        for i in range(nodes):
            temp = y_array[i] * baseLagrange(x_array, i, x_orginal[x])
            sum_ += temp
        result.append(sum_)
    return result


def rounding(array_x, array_y):
    node_x = []
    node_y = []
    for i in range(len(array_x)):
        node_x.append(round(array_x[i]))
        node_y.append(round(array_y[i]))
    return [node_x, node_x]


def loadData():
    x_values = []
    y_values = []
    # wczytywanie danych
    with open('road1.csv', mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            x, y = map(float, row)  # Konwertowanie wartości na float
            x_values.append(x)
            y_values.append(y)
    return [x_values, y_values]


def spline(x,x_array,result):
    z = 0
    y = 0
    function = []
    for i in range(len(x)):
        if x[i]>x_array[z+1]:
            z+=1
        y = result[z*4,0]+result[z*4+1,0]*x[i]+result[z*4+2,0]*x[i]**2+result[z*4+3,0]*x[i]**3
        function.append(y)
    return function



x_values, y_values = loadData()
size = len(x_values)
print(size)

n = 7  # liczba wezłow , stopień  wielomianu = n-1
distribution = math.floor(size / (n - 1))

array_x, array_y = linearDistribution(n, distribution)

# node_x,node_y = rounding()
# interpolationLagrangeR = interpolatedFunc(x_values,node_x,node_y,n)

interpolationLagrange = interpolatedFunc(x_values, array_x, array_y, n)

plt.figure(figsize=(15, 6))  # Ustawienie rozmiaru wykresu
plt.plot(x_values, y_values, color='b', label='original')
plt.plot(array_x, array_y, 'o', label='Nodes', color='r')  # 'o' oznacza marker punktu
plt.plot(x_values, interpolationLagrange, color='g', label='interpolation')
# plt.plot(x_values, interpolationLagrangeR, color='r', label='interpolationR')
for i in range(len(array_x)):
    plt.annotate(f'({array_x[i]:.2f}, {array_y[i]:.2f})', (array_x[i], array_y[i]), textcoords="offset points",
                 xytext=(0, 10), ha='center')
# plt.xlim(0, 4500)
# plt.xticks(range(0,25000, 4000))
plt.title('Interpolacja dla n węzłów')
plt.xlabel('Dystans [m]')
plt.ylabel('Wysokość [m]')
plt.legend()
plt.show()


def createMatrix(x,y,n):

    rows = 4 * n
    columns = rows
    matrix = np.zeros((rows, columns))
    vector = np.zeros((rows, 1))
    for i in range(n-1):
        h = x[i + 1] - x[i]  # h=xi+1-xi
        # a0 = f (x0)
        matrix[2 * i, 4 * i] = 1
        # a0 + b0h + c0h^2 + d0h^3 = f (x1)
        matrix[2 * i + 1, 4 * i] = 1
        matrix[2 * i + 1, 4 * i + 1] = h
        matrix[2 * i + 1, 4 * i + 2] = h ** 2
        matrix[2 * i + 1, 4 * i + 3] = h ** 3
        vector[2 * i, 0] = y[i]  # f (xi)
        vector[2 * i + 1, 0] = y[i + 1]  # f (xi+1)
    for i in range(2 * n, 3 * n - 1):
        index = i-2*n
        h = x[index + 1] - x[index]  # h=xi+1-xi
        matrix[i, 4 * index + 1] = 1
        matrix[i, 4 * index + 2] = 2 * h
        matrix[i, 4 * index + 3] = 3 * h ** 2
        matrix[i, 4 * index + 5] = -1
    for i in range(3 * n - 1, 4 * n - 2):
        index = i - 3 * n + 1
        h = x[index + 1] - x[index]  # h=xi+1-xi

        matrix[i, 4 * index + 2] = 2
        matrix[i, 4 * index + 3] = 6 * h
        matrix[i, 4 * index + 6] = -2
    h = x[n - 1] - x[n - 2]
    matrix[4 * n - 2, 2] = 2
    matrix[4 * n - 1, 4 * n - 2] = 2
    matrix[4 * n - 1, 4 * n - 1] = 6 * h

    return [matrix,vector]


#A,b = createMatrix(array_x, array_y,n)
A,b = createMatrix([1,3,5], [6,-2,4],3)
result = np.linalg.solve(A, b)
print(result)


splineInterpolation = spline(x_values,array_x,result)










