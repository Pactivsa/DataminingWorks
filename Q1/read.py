def read_data(data_path = ''):
    with open(data_path, 'r') as f:
        lines = f.readlines()
        # 数据以三个空格分隔    X   Y  
    