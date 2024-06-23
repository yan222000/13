% 给定参数
product_volumes = [400, 400, 400];  % 假设每种产品体积相同，简化处理
printer_speeds = [20, 40, 60];  % 打印机速度，单位mm/s
material_costs = [0.07, 0.1, 0.3];  % 打印材料成本，单位$/克
product_demand = [2000, 1000, 400];  % 每种产品级别的需求量
product_prices = [5, 10, 20];  % 产品级别的价格
T = 30;  % 可用总天数
downtimes = [8/24, 8/48, 8/72];  % 每台打印机的停机时间
available_times = (T * (1 - downtimes)) * 24 * 60 * 60;  % 可用秒数
% 变量数量
num_products = 3;
num_printers = 3;
 
% 目标函数系数（最小化负利润转换为最大化利润）
c = zeros(num_products * num_printers, 1);
for i = 1:num_products
    for j = 1:num_printers
        profit_per_unit_time = (product_prices(i) - material_costs(j)) * product_demand(i);
        c((i-1)*num_printers + j) = -profit_per_unit_time;  % 最小化负利润
    end
end
% 约束条件
% 每种产品的需求必须得到满足
A = zeros(num_products, num_products * num_printers);
b = zeros(num_products, 1);
for i = 1:num_products
    for j = 1:num_printers
        A(i, (i-1)*num_printers + j) = product_volumes(i) * printer_speeds(j);
    end
    b(i) = product_demand(i);
end
% 每台打印机的使用时间不得超过可用时间
for j = 1:num_printers
    for i = 1:num_products
        A(num_products + j, (i-1)*num_printers + j) = 1;
    end
    b(num_products + j) = available_times(j);
end
% 变量界限：所有时间必须非负
lb = zeros(num_products * num_printers, 1);
% 线性规划求解
options = optimoptions('linprog', 'Algorithm', 'dual-simplex', 'Display', 'iter');
[x, fval, exitflag, output] = linprog(c, A, b, [], [], lb, [], options);
% 显示结果
disp('最优解：');
disp(x);
disp('目标函数值（负利润）：');
disp(fval);


% 参数定义
T = 30;  % 总天数
q = [2000, 1000, 400];  % 产品需求量
v = pi * 2^2 * 10;  % 产品体积 (cm^3)
S = [20, 40, 60] * 3600;  % 打印速度 (cm^3/hr)
P = [1699, 2000, 3000];  % 打印机成本 ($/hr)
downtimes = [8/24, 8/48, 8/72];  % 停机时间比率
operational_hours = 24 * T * (1 - downtimes);  % 可用小时
 
% 构建线性规划参数
% 目标函数系数（成本最小化）
c = P;
 
% 约束矩阵和向量
% A*x <= b
% 需求满足约束转换为线性形式：x1*S1/v + x2*S2/v + x3*S3/v >= qi for each i
% 时间约束：xj <= operational_hours[j] for each j
A = zeros(length(q) + length(S), length(S));
b = zeros(length(q) + length(S), 1);
 
% 设置需求满足约束
for i = 1:length(q)
    A(i, :) = -S / v;  % 取负号因为要形成<=形式
    b(i) = -q(i);
end
 
% 设置时间约束
for j = 1:length(S)
    A(length(q) + j, j) = 1;
    b(length(q) + j) = operational_hours(j);
end
 
% 变量界限
lb = zeros(1, length(S));  % 所有打印机使用时间非负
ub = [];  % 无上界
 
% 求解线性规划问题
options = optimoptions('linprog', 'Algorithm', 'dual-simplex', 'Display', 'iter');
[x, fval, exitflag, output] = linprog(c, A, b, [], [], lb, ub, options);
 
% 输出结果
if exitflag == 1
    fprintf('最优打印机运行时间:\n');
    disp(x);
    fprintf('总成本: %.2f\n', fval);
else
    fprintf('优化失败: %s\n', output.message);
end


from scipy.optimize import minimize
import numpy as np
# 参数定义
T = 30  # 总天数
q = np.array([2000, 1000, 400])  # 产品需求量
v = np.pi * (2**2) * 10  # 产品体积
S = np.array([20, 40, 60]) * 3600  # 打印速度 (cm^3/hr)
downtimes = np.array([8/24, 8/48, 8/72])  # 停机时间比率
operational_hours = 24 * T * (1 - downtimes) * 3600  # 可用秒
# 定义目标函数
def objective(x):
    return max(x)  # 最小化最大使用时间
# 约束条件
def constraints(x):
    cons = []
    # 生产需求满足约束
    for k in range(len(q)):
        cons.append(sum(x[i * 2] * S[i] / v + x[i * 2 + 1] * S[i] / v for i in range(len(S))) - q[k])
    # 时间使用约束
    for i in range(len(S)):
        for j in range(2):  # 每种型号有两台打印机
            cons.append(x[i * 2 + j] - operational_hours[i])
    return cons
# 变量界限
bounds = [(0, operational_hours[i]) for i in range(len(S)) for _ in range(2)]
# 初始猜测
initial_guess = [operational_hours[i] / 2 for i in range(len(S)) for _ in range(2)]
# 使用非线性规划求解器
result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints={'type': 'ineq', 'fun': constraints})
# 输出结果
if result.success:
    print("Optimal printer operational hours:", result.x)
    print("Total maximum time used:", result.fun)
else:
    print("Optimization failed:", result.message)


