import os  # 导入os模块，用于处理文件路径
import pandas as pd  # 导入pandas库，用于数据处理
import numpy as np  # 导入numpy库，用于数值计算
from sklearn.model_selection import train_test_split  # 导入train_test_split函数，用于数据集划分
from sklearn.ensemble import RandomForestRegressor  # 导入随机森林回归模型
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error  # 导入均方误差评估指标
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import pickle  # 导入pickle模块，用于模型保存和加载
from sklearn.model_selection import KFold


# 统一输出路径
output_dir = r"E:\My_python\pythonProject\SSA-RF小论文12.4\SSA-RF"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class SSA:
    def __init__(self, func, bounds, population_size=30, iterations=50,random_seed=42):
        self.func = func  # 传入的评估函数
        self.bounds = bounds  # 参数搜索范围
        self.population_size = population_size  # 种群大小
        self.iterations = iterations  # 迭代次数
        self.dim = len(bounds)  # 参数维度
        self.best_score = float('inf')  # 初始最佳得分设为无穷大000
        self.best_pos = None  # 初始最佳位置为空
        self.output_dir = output_dir  # 使用全局路径
        np.random.seed(random_seed)  # 设置全局随机数种子
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.best_params_per_iteration = []  # 存储每一代的最佳参数

    def initialize(self):
        population = np.random.rand(self.population_size, self.dim)  # 随机初始化种群
        for i in range(self.dim):
            population[:, i] = population[:, i] * (self.bounds[i][1] - self.bounds[i][0]) + self.bounds[i][0]  # 将种群参数初始化到指定范围内
        return population

    def evaluate(self, population):
        fitness = np.apply_along_axis(self.func, 1, population)  # 对种群中的个体进行评估
        return fitness

    def update(self, population, fitness):
        best_index = np.argmin(fitness)  # 找到种群中最佳个体的索引
        best_score = fitness[best_index]  # 最佳个体的评估得分
        best_pos = population[best_index].copy()  # 最佳个体的位置
        return best_score, best_pos


    def run(self):
        population = self.initialize()  # 初始化种群
        fitness = self.evaluate(population)  # 评估种群
        self.best_score, self.best_pos = self.update(population, fitness)  # 更新最佳个体信息
        avg_oob_errors = []  # 用于存储每次迭代的平均 OOB Error
        for t in range(self.iterations):


            # 用于存储每代中的麻雀位置和MSE值
            iteration_data = []

            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            p = np.random.rand(self.population_size, self.dim)
            q = np.random.rand(self.population_size, self.dim)

            for i in range(self.population_size):
                if p[i][0] < 0.8:
                    if r2[i][0] < 0.5:
                        population[i] = self.best_pos + np.abs(population[i] - self.best_pos) * np.exp(t / self.iterations - 1)
                    else:
                        population[i] = population[i] + np.abs(population[i] - self.best_pos) * np.exp(t / self.iterations - 1)
                else:
                    if q[i][0] < 0.5:
                        population[i] = population[i] + np.random.randn() * np.abs(population[i] - self.best_pos)
                    else:
                        population[i] = self.best_pos + np.random.randn() * np.abs(population[i] - self.best_pos)

                for d in range(self.dim):
                    if population[i, d] < self.bounds[d][0]:
                        population[i, d] = self.bounds[d][0]
                    if population[i, d] > self.bounds[d][1]:
                        population[i, d] = self.bounds[d][1]

            fitness = self.evaluate(population)  # 更新种群评估
            # 获取当前代最佳个体信息并记录
            current_best_score, current_best_pos = self.update(population, fitness)
            self.best_params_per_iteration.append({
                'iteration': t + 1,
                'best_n_estimators': int(current_best_pos[0]),
                'best_max_depth': int(current_best_pos[1]),
                'best_min_samples_split': int(current_best_pos[2]),
                'best_min_samples_leaf': int(current_best_pos[3]),
                'best_OOB': current_best_score
            })
            # 计算平均 OOB Error 并记录
            avg_oob_error = np.mean(fitness)
            avg_oob_errors.append({'iteration': t + 1, 'avg_oob_error': avg_oob_error})
            # 记录当前种群中每个个体的参数值和对应的MSE值
            for i in range(self.population_size):
                iteration_data.append({
                    'n_estimators': int(population[i][0]),
                    'max_depth': int(population[i][1]),
                    'min_samples_split': int(population[i][2]),
                    'min_samples_leaf': int(population[i][3]),
                    'OOB': fitness[i]
                })
                # 保存当前代的所有麻雀及其对应的 OOB 误差值为 Excel 文件
                df = pd.DataFrame(iteration_data)
                file_path = os.path.join(self.output_dir, f'iteration_{t + 1}.xlsx')
                df.to_excel(file_path, index=False)


            if current_best_score < self.best_score:
                self.best_score = current_best_score
                self.best_pos = current_best_pos

            best_params_df = pd.DataFrame(self.best_params_per_iteration)
            best_params_file_path = os.path.join(self.output_dir, 'best_params_per_iteration.xlsx')
            best_params_df.to_excel(best_params_file_path, index=False)

            # 保存平均 OOB Error 到 Excel 文件
            avg_oob_errors_df = pd.DataFrame(avg_oob_errors)
            avg_oob_errors_file_path = os.path.join(self.output_dir, 'avg_oob_errors_per_iteration.xlsx')
            avg_oob_errors_df.to_excel(avg_oob_errors_file_path, index=False)

        return self.best_pos, self.best_score

# 读取数据
data = pd.read_csv('新副本 - 副本.csv',header=None)

# 将DataFrame转换为numpy数组
data_array = data.to_numpy()  # 使用滤波后的数据
X = data_array[:, 0:-1]  # 提取特征X，去除最后一列作为特征数据
Y = data_array[:, -1]  # 提取标签Y，最后一列作为目标变量

# 重采样数据
X_resampled, Y_resampled = resample(X, Y, replace=True, n_samples=len(X) * 2, random_state=42)

# 划分重取样后的数据集
scaler = MinMaxScaler()  # 初始化 MinMaxScaler
X_std_resampled = scaler.fit_transform(X_resampled)  # 归一化
x_train, x_test, y_train, y_test = train_test_split(X_std_resampled, Y_resampled, test_size=0.2, random_state=42)

# 保存 MinMaxScaler 的归一化模型参数
scaler_min = scaler.data_min_
scaler_scale = scaler.data_range_


def evaluate(params):
    n_estimators, max_depth, min_samples_split, min_samples_leaf = (
        int(params[0]), int(params[1]), int(params[2]), int(params[3])
    )

    # 启用袋外评分
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=int(min_samples_split),  # 确保是整数
        min_samples_leaf=int(min_samples_leaf),  # 确保是整数
        random_state=42,
        oob_score=True
    )

    rf.fit(x_train, y_train)  # 使用增样后的训练集

    # 使用袋外评分进行评估
    oob_error = 1 - rf.oob_score_  # 1 - oob_score表示袋外误差

    return oob_error

# 修改 SSA 中的搜索范围
bounds = [
(20, 1000),
     (2, 100),
     (2, 50),
     (1, 50)]

# 初始化SSA
ssa = SSA(evaluate, bounds, population_size=30, iterations=50)

# 运行 SSA 优化
best_params, best_score = ssa.run()
best_n_estimators, best_max_depth = int(best_params[0]), int(best_params[1])
best_min_samples_split, best_min_samples_leaf = int(best_params[2]), int(best_params[3])

print(f'Best n_estimators: {best_n_estimators}')
print(f'Best max_depth: {best_max_depth}')
print(f'Best min_samples_split: {best_min_samples_split}')
print(f'Best min_samples_leaf: {best_min_samples_leaf}')
print(f'Best OOB error: {best_score}')

# 使用最优参数训练模型
rf_best = RandomForestRegressor(
    n_estimators=best_n_estimators,
    max_depth=best_max_depth,
    min_samples_split=best_min_samples_split,
    min_samples_leaf=best_min_samples_leaf,
    random_state=42
)
rf_best.fit(x_train, y_train)


# 使用测试集进行预测并计算 R^2
y_pred_test = rf_best.predict(x_test)
r2_best = r2_score(y_test, y_pred_test)
mae_best = mean_absolute_error(y_test, y_pred_test)
mse_best = mean_squared_error(y_test, y_pred_test)  # 计算 MSE

print(f"模型的 R^2: {r2_best}")
print(f"模型的 MAE: {mae_best}")
print(f"模型的 MSE: {mse_best}")
# 保存 R^2 和预测结果到 Excel 文件
results_df = pd.DataFrame({
    'True Value': y_test,
    'Predicted Value': y_pred_test
})
results_df['R^2'] = r2_best
results_df['MAE'] = mae_best
results_df['MSE'] = mse_best  # 添加 MSE

output_path_R_MAE_MSE = os.path.join(output_dir, 'predictions_vs_true_values_with_r2_MAE_MSE.xlsx')
results_df.to_excel(output_path_R_MAE_MSE, index=False)

print(f"测试集的预测值、R^2、MAE 和 MSE 已保存到 {output_path_R_MAE_MSE} 文件中")

# 使用训练集进行预测并计算 R^2、MAE 和 MSE
y_pred_train = rf_best.predict(x_train)  # 使用训练集进行预测
r2_train = r2_score(y_train, y_pred_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
mse_train = mean_squared_error(y_train, y_pred_train)  # 计算 MSE

# 输出训练集的 R^2、MAE 和 MSE
print(f"训练集的 R^2: {r2_train}")
print(f"训练集的 MAE: {mae_train}")
print(f"训练集的 MSE: {mse_train}")

# 保存 R^2、MAE、MSE 和预测结果到 Excel 文件
results_df = pd.DataFrame({
    'True Value': y_train,
    'Predicted Value': y_pred_train
})
results_df['R^2'] = r2_train
results_df['MAE'] = mae_train
results_df['MSE'] = mse_train  # 添加 MSE

# 定义文件保存路径
output_path_R_MAE_MSE_train = os.path.join(output_dir, 'predictions_vs_true_values_train_with_r2_MAE_MSE.xlsx')
results_df.to_excel(output_path_R_MAE_MSE_train, index=False)

print(f"训练集的预测值、R^2、MAE 和 MSE 已保存到 {output_path_R_MAE_MSE_train} 文件中")


with open('best_random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_best, f)  # 将最优模型保存到文件中
print('模型已保存到 best_random_forest_model.pkl')  # 打印保存成功信息

# 获取特征的重要性
feature_importances = rf_best.feature_importances_

# 假设特征变量的名称为"Feature 1", "Feature 2", ... "Feature n"，可以根据您的特征实际名称进行调整
feature_names = ["底部含水层厚度", "底部含水层富水性", "底部含水层富水压", "底部黏土层厚度","基岩厚度","煤层倾角","煤层埋藏深度","松基比","松深比","开采高度","工作面长度","开采方法"]

# 创建DataFrame并保存到Excel
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
})

# 保存特征重要性到Excel文件
output_path = os.path.join(output_dir, 'feature_importances.xlsx')
feature_importance_df.to_excel(output_path, index=False)

print(f"特征变量重要性已保存到 {output_path} 文件中")



# 重采样数据
X_resampled1, Y_resampled1 = resample(X, Y, replace=True, n_samples=len(X) * 2, random_state=42)

X_std_resampled1 = scaler.fit_transform(X_resampled1)  # 归一化

# 使用最佳参数组合进行十折交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 初始化用于存储每折的预测结果和评估指标
cv_results = pd.DataFrame(columns=['Fold', 'True Value', 'Predicted Value'])
metrics = {'Fold': [], 'R²': [], 'MAE': [], 'MSE': []}

# 进行十折交叉验证
fold_number = 1
for train_index, val_index in kf.split(X_std_resampled1):
    X_train, X_val = X_std_resampled1[train_index], X_std_resampled1[val_index]
    y_train, y_val = Y_resampled1[train_index], Y_resampled1[val_index]

    # 使用最佳参数组合训练Lasso模型
    rf_best.fit(X_train, y_train)

    # 预测验证集
    y_pred_val = rf_best.predict(X_val)

    # 计算每折的 R²、MAE 和 MSE
    r2 = r2_score(y_val, y_pred_val)
    mae = mean_absolute_error(y_val, y_pred_val)
    mse = mean_squared_error(y_val, y_pred_val)

    # 将每折的预测结果添加到cv_results DataFrame中
    fold_results = pd.DataFrame({
        'Fold': fold_number,
        'True Value': y_val,
        'Predicted Value': y_pred_val
    })
    cv_results = pd.concat([cv_results, fold_results], ignore_index=True)

    # 将每折的评估指标添加到指标字典中
    metrics['Fold'].append(fold_number)
    metrics['R²'].append(r2)
    metrics['MAE'].append(mae)
    metrics['MSE'].append(mse)

    fold_number += 1

# 保存预测结果和指标到 Excel 文件
output_path_cv_results = os.path.join(output_dir, 'SSA-RF_cross_validation_results.xlsx')
with pd.ExcelWriter(output_path_cv_results) as writer:
    # 保存每折的预测结果
    cv_results.to_excel(writer, sheet_name='Predictions', index=False)

    # 将每折的评估指标转换为 DataFrame，并保存到同一文件中的不同工作表
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_excel(writer, sheet_name='Metrics', index=False)

    # 计算 R²、MAE 和 MSE 的平均值，并保存
    avg_metrics = pd.DataFrame({
        'Metric': ['R²', 'MAE', 'MSE'],
        'Average': [metrics_df['R²'].mean(), metrics_df['MAE'].mean(), metrics_df['MSE'].mean()]
    })
    avg_metrics.to_excel(writer, sheet_name='Average Metrics', index=False)

print(f"十折交叉验证的预测结果和评估指标已保存到 {output_path_cv_results}")


with open('rf_best_model_daogao.pkl', 'wb') as f:
    pickle.dump(rf_best, f)  # 将最优模型保存到文件中
print('模型已保存到 rf_best_model_daogao.pkl')  # 打印保存成功信息



