import numpy as np
def read_data(data_path = ''):
    #data二维数组，存储所有点的坐标
    data = []
    with open(data_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # 以空格分割,并转成float
            # line = line.strip().split('    ')
            line = line.strip().split('    ')
            line = [float(i) for i in line]
            data.append(line)
            
    return np.array(data)

if __name__ == '__main__':
    data_path = "Q1/8gau.txt"
    data = read_data(data_path)
    print(data)
    
    