import pandas as pd
import matplotlib.pyplot as plt

# Excel文件的路径
file_path = './result/all_results.xlsx'

# 从Excel文件中加载结果
all_results = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')

# 创建一个图形用于绘制图表
plt.figure(figsize=(15, 10))

# 遍历每个算法的结果
for algo_name, result_df in all_results.items():
    # 提取RMSE、MAE、Fit Time和Test Time列
    result_df = result_df.dropna()  # 删除包含NaN值的行（均值和标准差）

    # 绘制RMSE和MAE图表
    plt.subplot(2, 2, 1)
    plt.plot(result_df.index, result_df['RMSE'], label=algo_name)
    plt.title('每个算法的RMSE')
    plt.xlabel('折数')
    plt.ylabel('RMSE')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(result_df.index, result_df['MAE'], label=algo_name)
    plt.title('每个算法的MAE')
    plt.xlabel('折数')
    plt.ylabel('MAE')
    plt.legend()

    # 绘制Fit Time和Test Time图表
    plt.subplot(2, 2, 3)
    plt.plot(result_df.index, result_df['Fit time'], label=algo_name)
    plt.title('每个算法的训练时间')
    plt.xlabel('折数')
    plt.ylabel('训练时间 (秒)')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(result_df.index, result_df['Test time'], label=algo_name)
    plt.title('每个算法的测试时间')
    plt.xlabel('折数')
    plt.ylabel('测试时间 (秒)')
    plt.legend()

# 调整布局并显示图表
plt.tight_layout()
plt.show()