import matplotlib.pyplot as plt
import numpy as np
import csv
import math


def linearDistribution(n,distribution):
    array_x = []
    array_y = []
    for i in range(n - 1):
        array_x.append(x_values[i * distribution])
        array_y.append(y_values[i * distribution])
    array_x.append(x_values[-1])
    array_y.append(y_values[-1])
    return [array_x,array_y]
def baseLagrange(x_array,index,x):
    counter = 1
    denominator = 1
    for i in range(len(x_array)):
        if i != index:
            counter *= x-x_array[i]
            denominator *= x_array[index] - x_array[i]

    return counter/denominator

def interpolatedFunc(x_orginal,x_array,y_array,nodes):
    result = []
    for x in range(len(x_orginal)):
        sum_ = 0
        for i in range(nodes):
            temp = y_array[i]*baseLagrange(x_array,i,x_orginal[x])
            sum_ += temp
        result.append(sum_)
    return result


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
    return [x_values,y_values]


x_values,y_values = loadData()
size = len( x_values)
print(size)

n = 5#liczba wezłow , stopień  wielomianu = n-1
distribution = math.floor( size/(n-1))

array_x,array_y = linearDistribution(n,distribution)

interpolationLagrange = interpolatedFunc(x_values,array_x,array_y,n)

plt.figure(figsize=(15, 6))  # Ustawienie rozmiaru wykresu
plt.plot(x_values, y_values, color='b', label='original')
plt.plot(array_x, array_y, color='r', label='points')
plt.plot(x_values, interpolationLagrange, color='g', label='interpolation')
plt.title('Wykres')
plt.xlabel('Dystans [m]')
plt.ylabel('Wysokość [m]')
plt.legend()
plt.show()




