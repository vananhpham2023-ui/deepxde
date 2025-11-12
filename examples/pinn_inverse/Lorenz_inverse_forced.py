"""Backend supported: tensorflow.compat.v1, tensorflow, paddle

物理信息神经网络(PINN)求解带外部输入的Lorenz吸引子参数辨识问题
本代码展示了如何使用DeepXDE库实现PINN方法来求解逆问题，即从观测数据中反演系统参数
see https://github.com/lululxvi/deepxde/issues/79
"""
import re

# 导入必要的库
import deepxde as dde  # 导入DeepXDE库，这是一个用于物理信息神经网络的Python库
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt  # 用于绘图
import numpy as np  # 用于数值计算
import scipy as sp  # 用于科学计算
from scipy.integrate import odeint  # 用于求解常微分方程

# 让 Matplotlib 使用支持中文的字体，避免中文显示为方框
_noto_font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
try:
    fm.fontManager.addfont(_noto_font_path)
    matplotlib.rcParams["font.family"] = fm.FontProperties(fname=_noto_font_path).get_name()
except FileNotFoundError:
    pass
matplotlib.rcParams["axes.unicode_minus"] = False


# 第一部分：生成训练数据
# 系统参数真实值，参考论文 https://arxiv.org/abs/1907.04502 第15页
C1true = 10  # Lorenz系统的第一个参数真实值
C2true = 15  # Lorenz系统的第二个参数真实值
C3true = 8 / 3  # Lorenz系统的第三个参数真实值

# 时间点设置
maxtime = 3  # 最大时间
# 在[0, maxtime]范围内生成200个等间隔的时间点
time = np.linspace(0, maxtime, 200)
# 生成外部输入信号：正弦波
ex_input = 10 * np.sin(2 * np.pi * time)  # 外部激励信号

# 定义外部输入的插值函数，用于在任意时间点获取外部输入值
def ex_func(t):
    """
    外部输入信号的插值函数
    使用径向基函数(RBF)插值，确保在任意时间点都能获取外部输入值
    这对于PINN中处理非固定时间步长的情况非常重要
    
    参数:
        t: 时间点或时间点数组
    返回:
        对应时间点的外部输入值
    """
    # 创建RBF插值器，使用thin_plate函数
    spline = sp.interpolate.Rbf(
        time, ex_input, function="thin_plate", smooth=0, episilon=0
    )
    return spline(t)


# 定义修改后的Lorenz系统（带外部输入）
def LorezODE(x, t):
    """
    修改后的Lorenz系统微分方程
    这是一个带外部输入的三阶非线性常微分方程组
    
    参数:
        x: 状态向量 [x1, x2, x3]
        t: 时间
    返回:
        状态导数向量 [dx1/dt, dx2/dt, dx3/dt]
    """
    x1, x2, x3 = x  # 解包状态向量
    # 计算每个状态的导数
    dxdt = [
        C1true * (x2 - x1),  # dx1/dt = C1*(x2 - x1)
        x1 * (C2true - x3) - x2,  # dx2/dt = x1*(C2 - x3) - x2
        x1 * x2 - C3true * x3 + ex_func(t),  # dx3/dt = x1*x2 - C3*x3 + 外部输入
    ]
    return dxdt


# 初始条件
x0 = [-8, 7, 27]  # Lorenz系统的初始状态

# 使用scipy的odeint求解常微分方程，生成训练数据
x = odeint(LorezODE, x0, time)  # x的形状为(200, 3)，每一行代表一个时间点的状态

# 绘制生成的数据，观察系统行为
plt.plot(time, x, time, ex_input)
plt.xlabel("时间")
plt.ylabel("状态值x(t)")
plt.show()

# 将时间数组重塑为二维数组，以适应DeepXDE的输入格式
time = time.reshape(-1, 1)  # 形状变为(200, 1)


# 第二部分：使用PINN进行参数辨识
# 定义待辨识的参数变量，初始值设为1.0
C1 = dde.Variable(1.0)  # 对应C1true，将通过PINN优化确定
C2 = dde.Variable(1.0)  # 对应C2true，将通过PINN优化确定
C3 = dde.Variable(1.0)  # 对应C3true，将通过PINN优化确定

# 定义用于PINN的外部输入插值函数
# 注意这里的输入格式与ex_func不同，需要处理二维数组
def ex_func2(t):
    """
    为PINN模型准备的外部输入插值函数
    与ex_func不同，此函数接收的是二维数组输入(t[:, 0:])
    这是因为DeepXDE的PDE数据对象在调用auxiliary_var_function时传入的是二维数组
    
    参数:
        t: 二维时间数组，形状为(N, 1)
    返回:
        对应时间点的外部输入值
    """
    spline = sp.interpolate.Rbf(
        time[:, 0], ex_input, function="thin_plate", smooth=0, episilon=0
    )
    return spline(t[:, 0:])  # t[:, 0:] 确保我们获取所有行的第一个元素


# 定义Lorenz系统的PDE残差函数
def Lorenz_system(x, y, ex):
    """
    定义修改后的Lorenz系统的PDE残差
    在PINN中，我们将微分方程转换为残差形式：残差 = 导数 - 右侧表达式
    
    参数:
        x: 输入变量（这里是时间t）
        y: 神经网络的输出（状态变量[x1, x2, x3]）
        ex: 外部输入信号
    返回:
        残差列表，每个残差对应一个微分方程
    """
    # 解包神经网络输出的三个状态分量
    y1, y2, y3 = y[:, 0:1], y[:, 1:2], y[:, 2:]
    
    # 使用DeepXDE的自动微分功能计算各状态的时间导数
    # dde.grad.jacobian(y, x, i)计算y的第i个分量对x的导数
    dy1_x = dde.grad.jacobian(y, x, i=0)  # dx1/dt
    dy2_x = dde.grad.jacobian(y, x, i=1)  # dx2/dt
    dy3_x = dde.grad.jacobian(y, x, i=2)  # dx3/dt
    
    # 计算每个方程的残差
    # 残差 = 计算的导数 - 理论导数表达式
    return [
        dy1_x - C1 * (y2 - y1),  # 第一个方程的残差
        dy2_x - y1 * (C2 - y3) + y2,  # 第二个方程的残差（注意符号修正）
        dy3_x - y1 * y2 + C3 * y3 - ex,  # 第三个方程的残差
    ]


# 定义边界条件：初始条件的判断函数
def boundary(_, on_initial):
    """
    边界条件判断函数
    用于确定哪些点是初始点
    
    参数:
        _: 输入点（在此例中未使用）
        on_initial: 是否在初始边界上
    返回:
        布尔值，表示是否是初始条件点
    """
    return on_initial


# 定义时间域
# TimeDomain是DeepXDE中用于定义一维时间几何区域的类
# 位于deepxde/geometry/timedomain.py
geom = dde.geometry.TimeDomain(0, maxtime)  # 创建[0, maxtime]的时间域

# 定义初始条件
# IC类用于定义初始条件，位于deepxde/icbc/__init__.py
# 参数分别为：几何区域、初始条件函数、边界判断函数、对应的状态分量
ic1 = dde.icbc.IC(geom, lambda X: x0[0], boundary, component=0)  # x1的初始条件
ic2 = dde.icbc.IC(geom, lambda X: x0[1], boundary, component=1)  # x2的初始条件
ic3 = dde.icbc.IC(geom, lambda X: x0[2], boundary, component=2)  # x3的初始条件

# 准备观测数据作为边界条件约束
observe_t, ob_y = time, x  # 观测时间和对应的观测状态
# PointSetBC类用于定义点集边界条件，将观测数据作为约束条件
observe_y0 = dde.icbc.PointSetBC(observe_t, ob_y[:, 0:1], component=0)  # x1的观测约束
observe_y1 = dde.icbc.PointSetBC(observe_t, ob_y[:, 1:2], component=1)  # x2的观测约束
observe_y2 = dde.icbc.PointSetBC(observe_t, ob_y[:, 2:3], component=2)  # x3的观测约束

# 创建PDE数据对象
# PDE类位于deepxde/data/pde.py，是DeepXDE处理PDE问题的核心数据结构
# 参数说明：
# - geom: 几何区域
# - Lorenz_system: PDE残差函数
# - [ic1, ic2, ic3, observe_y0, observe_y1, observe_y2]: 边界条件和约束条件列表
# - num_domain: 域内采样点数量
# - num_boundary: 边界采样点数量
# - anchors: 额外的固定点（这里是观测数据点）
# - auxiliary_var_function: 辅助变量函数（这里是外部输入）
data = dde.data.PDE(
    geom,
    Lorenz_system,
    [ic1, ic2, ic3, observe_y0, observe_y1, observe_y2],
    num_domain=400,  # 域内随机采样400个点用于训练
    num_boundary=2,  # 边界采样点数量
    anchors=observe_t,  # 使用观测时间点作为固定锚点
    auxiliary_var_function=ex_func2,  # 提供外部输入的函数
)

# 绘制训练数据，确认数据正确性
plt.plot(observe_t, ob_y)
plt.xlabel("时间")
plt.legend(["x", "y", "z"])
plt.title("训练数据")
plt.show()


# 第三部分：定义神经网络模型并训练
# 定义前馈神经网络(FNN)架构
# FNN类位于deepxde/nn/__init__.py
# [1] + [40] * 3 + [3] 表示网络结构：1个输入节点，3个隐藏层每层40个节点，3个输出节点
# "tanh"是激活函数，"Glorot uniform"是权重初始化方法
net = dde.nn.FNN([1] + [40] * 3 + [3], "tanh", "Glorot uniform")

# 创建PINN模型
# Model类位于deepxde/model.py，是DeepXDE的核心模型类
# 它将数据和网络连接起来，处理训练过程
model = dde.Model(data, net)

# 编译模型
# compile方法配置优化器、学习率和其他训练参数
# external_trainable_variables指定外部可训练变量（即我们要辨识的参数）
model.compile("adam", lr=0.001, external_trainable_variables=[C1, C2, C3])

# 创建回调函数，用于保存训练过程中的参数值
# VariableValue回调位于deepxde/callbacks.py
# 它会定期保存指定变量的值到文件中
fnamevar = "variables.dat"  # 保存参数值的文件名
variable = dde.callbacks.VariableValue([C1, C2, C3], period=100, filename=fnamevar)

# 训练模型
# train方法执行训练过程，参数包括迭代次数和回调函数
model.train(iterations=60000, callbacks=[variable])


# 第四部分：结果分析和可视化
# 从文件中读取保存的参数值
lines = open(fnamevar, "r").readlines()
# 解析文件内容，提取参数值
# 这行代码使用正则表达式从每行中提取方括号内的数值
Chat = np.array(
    [
        np.fromstring(
            min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
            sep=",",
        )
        for line in lines
    ]
)

# 绘制参数收敛过程
l, c = Chat.shape  # 获取参数历史的形状
plt.figure()
plt.plot(range(l), Chat[:, 0], "r-")  # 绘制C1的收敛过程
plt.plot(range(l), Chat[:, 1], "k-")  # 绘制C2的收敛过程
plt.plot(range(l), Chat[:, 2], "g-")  # 绘制C3的收敛过程
# 绘制真实值的水平线作为参考
plt.plot(range(l), np.ones(Chat[:, 0].shape) * C1true, "r--")
plt.plot(range(l), np.ones(Chat[:, 1].shape) * C2true, "k--")
plt.plot(range(l), np.ones(Chat[:, 2].shape) * C3true, "g--")
plt.legend(["C1hat", "C2hat", "C3hat", "真实C1", "真实C2", "真实C3"], loc="right")
plt.xlabel("迭代次数")
plt.title("参数收敛过程")
plt.show()

# 使用训练好的模型进行预测
yhat = model.predict(observe_t)

# 绘制预测结果与真实值的比较
plt.figure()
plt.plot(observe_t, ob_y, "-", observe_t, yhat, "--")  # 实线表示真实值，虚线表示预测值
plt.xlabel("时间")
plt.legend(["x真实值", "y真实值", "z真实值", "x预测值", "y预测值", "z预测值"])
plt.title("PINN预测结果与真实值比较")
plt.show()

# 打印最终辨识的参数值
print("\n最终辨识的参数值：")
print(f"C1 = {float(C1):.6f} (真实值: {C1true})")
print(f"C2 = {float(C2):.6f} (真实值: {C2true})")
print(f"C3 = {float(C3):.6f} (真实值: {C3true})")

# 打印相对误差
print("\n参数辨识相对误差：")
print(f"C1相对误差: {abs(float(C1) - C1true) / C1true * 100:.4f}%")
print(f"C2相对误差: {abs(float(C2) - C2true) / C2true * 100:.4f}%")
print(f"C3相对误差: {abs(float(C3) - C3true) / C3true * 100:.4f}%")
