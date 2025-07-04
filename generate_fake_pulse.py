import matplotlib.pyplot as plt
import numpy as np
from random import randrange

mode = 'random'
ordered_it = 0

def increment():
    global ordered_it
    ordered = [5,1,2,3,3,4,0,1]
    match mode:
        case 'even':
            return 50
        case 'random':
            return randrange(10)*10
        case 'ordered':
            i =  ordered[ordered_it]*10
            if ordered_it+1 == len(ordered):
                ordered_it = 0
            else:
                ordered_it += 1
            return i
    return 0



def create_square_wave():
    x = np.full(3500, -1.0)
    n = 0
    while n < 3500:
        n += increment()
        x[n:n+50] = 1
        n += 50
    return x


def create_triangle_wave():
    x = np.full(3500, -1.0)
    n = 0
    width = 50
    while n + width < 3500:
        n += increment()
        if n + width/2 >= 3500:
            break
        up_slope = np.linspace(-1, 1, int(width/2))
        x[n:int(n+width/2)] = up_slope
        n += int(width/2)

        if n + width/2 >= 3500:
            break
        down_slope = np.linspace(1, -1, int(width/2))
        x[n:int(n+width/2)] = down_slope
        n += int(width/2)
    return x


square_wave = np.array([create_square_wave() for _ in range(10000)])
fig = plt.figure(figsize=(15,4))
plt.plot(square_wave[0])
plt.show()
fig = plt.figure(figsize=(15,4))
plt.plot(square_wave[10])
plt.show()
np.savetxt(f'processed_data/square_wave_{mode}.csv', square_wave, delimiter=',')


triangle_wave = np.array([create_triangle_wave() for _ in range(10000)])
fig = plt.figure(figsize=(15,4))
plt.plot(triangle_wave[0])
plt.show()
fig = plt.figure(figsize=(15,4))
plt.plot(triangle_wave[0])
plt.show()
np.savetxt(f'processed_data/triangle_wave_{mode}.csv', triangle_wave, delimiter=',')
