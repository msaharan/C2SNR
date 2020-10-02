import multiprocessing as mp
import numpy as np
import time

def mp_a(i, j):
    #    a[i, j] = i + j
    #return a[i, j]
    return i + j

def mp_b(k, l):
    b[k, l] = k + l
    return b[k, l]


"""
a = np.zeros((100, 100))
processes_a = []
starttime_a = time.time()

for i in range(0, 100):
    for j in range(0, 100):
        p_a = mp.Process(target = mp_a, args = (i,j))
        processes_a.append(p_a)
        p_a.start()

for process in processes_a:
        process.join()
"""

print('hello')
hello = 1

a = np.zeros(10)

def mp_a_s(i):
    a[i] = i
    hello = i
#    print(hello)
#    print(a)


if __name__ == '__main__':

    limit = 10

    processes_a = []
    starttime_a = time.time()
    print(hello)
    for i in range(0, limit):

        p_a = mp.Process(target = mp_a_s, args = (i,))
#        processes_a.append(p_a)
        p_a.start()
        p_a.join()



#    for process in processes_a:
#        process.join()

    print('a took {} seconds'.format(time.time() - starttime_a))

print(hello)
"""
for k in range(0, 100):
    for l in range(0, 100):
        mp_b(k, l)
"""     

