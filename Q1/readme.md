# 高斯混合模型

[高斯混合模型](GMM.py)定义
使用方法
```python

    from GMM import GMM # 导入GMM类
    gmm = GMM(n_components=3) # 初始化GMM模型
    gmm.fit(X) # sklearn风格的训练方法
    y = gmm.predict(X) # sklearn风格的预测方法

```

