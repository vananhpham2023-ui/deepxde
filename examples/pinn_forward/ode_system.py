"""
常微分方程组求解示例 - PINN方法实现
"""
# 导入必要的库
import deepxde as dde  # 导入DeepXDE库，这是一个用于求解微分方程的深度学习库
import numpy as np      # 导入NumPy库用于数值计算

"""
    定义常微分方程组：
    dy1/dx = y2
    dy2/dx = -y1
    参数: x: 自变量；y: 神经网络的输出，是一个包含两个分量的张量 [y1, y2]
    返回:方程组的残值列表：[dy1/dx - y2, dy2/dx + y1]
    注意：
        1. 与单个ODE不同，此处神经网络输出两个分量，分别代表y1和y2
        2. 代码包含了不同后端的实现注释，特别是JAX后端的特殊处理
        3. dde.grad.jacobian函数的i参数指定了要计算哪个输出分量的导数
"""
def ode_system(x, y):

    # 将网络输出分离为两个分量y1和y2
    y1, y2 = y[:, 0:1], y[:, 1:]
    dy1_x = dde.grad.jacobian(y, x, i=0)  # i=0表示对第一个输出分量求导
    dy2_x = dde.grad.jacobian(y, x, i=1)  # i=1表示对第二个输出分量求导
    
    # JAX后端的特殊实现（当前被注释掉）
    # y_val, y_fn = y
    # y1, y2 = y_val[:, 0:1], y_val[:, 1:]
    # dy1_x, _ = dde.grad.jacobian(y, x, i=0)
    # dy2_x, _ = dde.grad.jacobian(y, x, i=1)
    
    return [dy1_x - y2, dy2_x + y1]

#    定义边界条件的位置
def boundary(_, on_initial):
    return on_initial

#定义解析解函数，返回方程组的精确解：[sin(x), cos(x)]
def func(x):
    # 使用np.hstack将两个解向量水平堆叠
    return np.hstack((np.sin(x), np.cos(x)))


# 创建时间域几何对象，指定求解区间为x ∈ [0, 10]，TimeDomain类位于deepxde/gemetry/timedomain.py中
gem = dde.gemetry.TimeDomain(0, 10)

# 创建第一个初始条件对象（对应y1(0) = 0）
# IC类位于deepxde/icbc/__init__.py中
# component=0指定这是针对输出的第一个分量y1的初始条件
ic1 = dde.icbc.IC(gem, lambda x: 0, boundary, component=0)

# 创建第二个初始条件对象（对应y2(0) = 1）
# component=1指定这是针对输出的第二个分量y2的初始条件
ic2 = dde.icbc.IC(gem, lambda x: 1, boundary, component=1)

# 创建PDE数据对象
# PDE类位于deepxde/data/pde.py中，用于设置偏微分方程问题
# 参数说明：
# - gem: 几何区域
# - ode_system: 定义的ODE方程组
# - [ic1, ic2]: 初始条件列表
# - 35: 在域内采样的训练点数量
# - 2: 在边界上采样的训练点数量
# - solution=func: 提供解析解用于计算误差
# - num_test=100: 用于测试的点数量
data = dde.data.PDE(gem, ode_system, [ic1, ic2], 35, 2, solution=func, num_test=100)

# 定义神经网络结构
# 网络结构：1个输入层(1维) + 3个隐藏层(每层50个神经元) + 1个输出层(2维)
# 注意：输出层维度为2，因为我们需要同时预测y1和y2
layer_size = [1] + [50] * 3 + [2]

# 选择激活函数为tanh
activation = "tanh"

# 选择权重初始化方法为Glorot uniform (Xavier初始化)
initializer = "Glorot uniform"

# 创建全连接神经网络(FNN)模型
# FNN类根据当前使用的后端从对应目录加载实现
# 例如，使用PyTorch后端时从deepxde/nn/pytorch/fnn.py加载
net = dde.nn.FNN(layer_size, activation, initializer)

# 创建DeepXDE模型
# Model类位于deepxde/model.py中，是整个框架的核心类
# 它将数据、网络和损失函数组合在一起
model = dde.Model(data, net)

# 编译模型
# 设置优化器为Adam，学习率为0.001
# metrics设置为l2相对误差，用于评估模型性能
# 对于方程组问题，损失函数会自动处理两个方程的残值
model.compile("adam", lr=0.001, metrics=["l2 relative error"])

# 训练模型
# iterations参数指定训练轮数（这里设置为20000，比单个ODE多，因为方程组更复杂）
# 返回值：
# - losshistory: 训练过程中的损失记录
# - train_state: 训练后的模型状态
losshistory, train_state = model.train(iterations=20000)

# 保存并绘制训练结果
# saveplot函数位于deepxde/display.py中
# issave=True: 保存结果到文件
# isplot=True: 显示结果图像
# 对于方程组，将分别绘制y1和y2的预测结果与解析解的比较
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
