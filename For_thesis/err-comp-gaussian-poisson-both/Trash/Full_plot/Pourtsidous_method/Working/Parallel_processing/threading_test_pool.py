from multiprocessing import Pool
import numpy as np

def mp_a_s(i):
    return i

data = np.zeros(10)
print(data)
if __name__ == '__main__':
    p = Pool(processes = 20000)
    data = p.map(mp_a_s, [i for i in range(0,20000)])
    p.close()
    print(data + data)



