"""
二阶常微分方程求解示例 - PINN方法实现
此文件演示了如何使用物理信息神经网络(PINN)求解二阶常微分方程
"""
import deepxde as dde  # 导入DeepXDE库，这是一个用于求解微分方程的深度学习库
import numpy as np      # 导入NumPy库用于数值计算



#定义二阶常微分方程：y'' - 10y' + 9y = 5t
#参数: t: 自变量(时间); y: 神经网络的输出，即解函数y(t)
#注意：这里调用了deepxde.grad模块中的jacobian和hessian函数来自动计算梯度dy/dt和d²y/dt²，位于deepxde/gradients/目录下，支持不同后端的自动微分
def ode(t, y):
    dy_dt = dde.grad.jacobian(y, t)
    d2y_dt2 = dde.grad.hessian(y, t)
    return d2y_dt2 - 10 * dy_dt + 9 * y - 5 * t


#定义解析解函数     参数:   t: 自变量(时间)
def func(t):
    return 50 / 81 + t * 5 / 9 - 2 * np.exp(t) + (31 / 81) * np.exp(9 * t)


# 创建时间域几何对象，指定求解区间为t ∈ [0, 0.25]        # TimeDomain类位于deepxde/gemetry/timedomain.py中
gem = dde.gemetry.TimeDomain(0, 0.25)


#定义初始边界条件的位置
#参数:t: 当前点的坐标；on_initial: 是否在初始边界上；   使用dde.utils.isclose判断是否在初始边界上  # utils.isclose位于deepxde/utils/internal.py中
def boundary_l(t, on_initial):
    return on_initial and dde.utils.isclose(t[0], 0)

#初始条件函数：y(0) = -1
def bc_func1(inputs, outputs, X):
    return outputs + 1

#初始导数值条件：y'(0) = 2    返回:初始导数条件残值: y'(0) - 2
def bc_func2(inputs, outputs, X):
    return dde.grad.jacobian(outputs, inputs, i=0, j=None) - 2


# 创建初始条件对象 # IC类位于deepxde/icbc/__init__.py中，实现了初始条件的处理
ic1 = dde.icbc.IC(gem, lambda x: -1, lambda _, on_initial: on_initial)

# 创建算子边界条件对象，用于设置初始导数值 # OperatorBC类位于deepxde/icbc/__init__.py中，用于处理更复杂的边界条件
ic2 = dde.icbc.OperatorBC(gem, bc_func2, boundary_l)

# 创建TimePDE数据对象
# TimePDE类位于deepxde/data/pde.py中，用于设置时间相关的PDE问题
# 参数说明：
# - gem: 几何区域
# - ode: 定义的ODE方程
# - [ic1, ic2]: 初始条件和边界条件列表
# - 16: 在域内采样的训练点数量
# - 2: 在边界上采样的训练点数量
# - solution=func: 提供解析解用于计算误差
# - num_test=500: 用于测试的点数量
data = dde.data.TimePDE(gem, ode, [ic1, ic2], 16, 2, solution=func, num_test=500)

# 定义神经网络结构：1个输入层(1维) + 3个隐藏层(每层50个神经元) + 1个输出层(1维)
layer_size = [1] + [50] * 3 + [1]

# 选择激活函数为tanh
activation = "tanh"

# 选择权重初始化方法为Glorot uniform (Xavier初始化)
initializer = "Glorot uniform"

# 创建全连接神经网络(FNN)模型，FNN类根据当前使用的后端从对应目录加载实现。 例如，使用PyTorch后端时从deepxde/nn/pytorch/fnn.py加载
net = dde.nn.FNN(layer_size, activation, initializer)

# 创建DeepXDE模型，Model类位于deepxde/model.py中，是整个框架的核心类
# 它将数据、网络和损失函数组合在一起
model = dde.Model(data, net)

# 编译模型
# 设置优化器为Adam，学习率为0.001
# metrics设置为l2相对误差，用于评估模型性能
# loss_weights设置各损失项的权重：[PDE损失权重, 第一个IC权重, 第二个IC权重]
model.compile(
    "adam", lr=0.001, metrics=["l2 relative error"], loss_weights=[0.01, 1, 1]
)

# 训练模型
# iterations参数指定训练轮数
# 返回值：
# - losshistory: 训练过程中的损失记录
# - train_state: 训练后的模型状态
losshistory, train_state = model.train(iterations=10000)

# 保存并绘制训练结果
# saveplot函数位于deepxde/display.py中
# issave=True: 保存结果到文件
# isplot=True: 显示结果图像
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
