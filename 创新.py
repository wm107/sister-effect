import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind, norm

# ====================== 1. 读取数据 ======================
baseline_stu = pd.read_excel('基线学生.xlsx')
baseline_par = pd.read_excel('基线家长.xlsx')
follow_stu = pd.read_excel('追加学生数据.xlsx')
follow_par = pd.read_excel('追加家长数据.xlsx')

# ====================== 2. 合并数据 ======================
baseline_data = pd.merge(baseline_stu, baseline_par, on='ids', how='inner')
follow_data = pd.merge(follow_stu, follow_par, on='ids', how='inner')
data_raw = pd.concat([baseline_data, follow_data], ignore_index=True).drop_duplicates(subset='ids')

# ====================== 3. 核心统计指标预计算（新增模块） ======================
# 1. 全部原始总样本量
total_raw_sample = len(data_raw)
# 2. 独生子女样本量
only_child_sample = (data_raw['b01'] == 1).sum()

# 只保留非独生子女（原有逻辑）
data = data_raw[data_raw['b01'] == 2].copy()

# ====================== 4. 计算同胞数量 ======================
data['total_siblings'] = data[['b0201', 'b0202', 'b0203', 'b0204']].sum(axis=1)
# 剔除 total_siblings=0 但 b01=2 的样本（自称非独生但未填写兄弟姐妹信息）
data = data[data['total_siblings'] >= 1].copy()
print(f"剔除矛盾样本后剩余样本量: {len(data)}")

# ====================== 5. 核心变量定义 ======================
data['has_sister'] = np.where(data['b0203'] >= 1, 1, 0)
data['has_brother'] = np.where(data['b0201'] >= 1, 1, 0)
data['two_child_only_sister'] = (
        (data['total_siblings'] == 1) & (data['b0203'] == 1) &
        (data['b0201'] == 0) & (data['b0202'] == 0) & (data['b0204'] == 0)
).astype(int)
data['multi_with_sister'] = ((data['has_sister'] == 1) & (data['total_siblings'] >= 2)).astype(int)
data['no_sister'] = (data['has_sister'] == 0).astype(int)
data['is_multi'] = (data['total_siblings'] >= 2).astype(int)
data['sister_multi'] = data['has_sister'] * data['is_multi']

# 多孩家庭内部细分
data['only_sister_multi'] = ((data['multi_with_sister'] == 1) & (data['b0201'] == 0)).astype(int)
data['sister_and_brother'] = ((data['multi_with_sister'] == 1) & (data['b0201'] >= 1)).astype(int)

# ====================== 6. 控制变量 ======================
data['par_edu'] = (data['stfedu'] + data['stmedu']) / 2
data['hukou'] = np.where(data['sthktype'] == 1, 1, 0)
data['gender'] = data['stsex']

# ====================== 7. 标准化成绩 ======================
subjects = ['tr_chn', 'tr_mat', 'tr_eng']
scaler = StandardScaler()
data[['std_chn', 'std_mat', 'std_eng']] = scaler.fit_transform(data[subjects])

# ====================== 8. 统一筛选：只保留三科成绩都齐全的样本 ======================
data = data.dropna(subset=['std_chn', 'std_mat', 'std_eng']).copy()
print(f"统一筛选后样本量: {len(data)}")

# 然后计算 GPA（此时三科都不缺失）
data['gpa'] = data[['std_chn', 'std_mat', 'std_eng']].mean(axis=1)

# ====================== 9. 清理其他控制变量缺失 ======================
data = data.dropna(subset=['has_sister', 'gender', 'par_edu', 'hukou', 'total_siblings']).copy()
print(f"最终有效样本量: {len(data)}")
# ====================== 9. 生成你需要的样本统计Excel表（核心新增模块） ======================
# 计算所有需要的指标
total_valid_sample = len(data)  # 最终有效分析样本量
two_child_total = (data['total_siblings'] == 1).sum()  # 二孩家庭总样本
two_child_has_sister = (data['two_child_only_sister'] == 1).sum()  # 二孩家庭有姐姐的样本
two_child_sister_ratio = (two_child_has_sister / two_child_total * 100) if two_child_total > 0 else 0  # 二孩有姐姐占比

multi_child_total = (data['total_siblings'] >= 2).sum()  # 多孩家庭总样本
multi_child_has_sister = (data['multi_with_sister'] == 1).sum()  # 多孩家庭有姐姐的样本
multi_child_sister_ratio = (multi_child_has_sister / multi_child_total * 100) if multi_child_total > 0 else 0  # 多孩有姐姐占比

# 构建统计表格
sample_stats_df = pd.DataFrame([
    {
        '统计指标': '全部有效分析样本量',
        '样本数量': total_valid_sample,
        '占比(%)': 100.00,
        '备注': '最终用于回归分析的干净样本'
    },
    {
        '统计指标': '原始合并总样本量',
        '样本数量': total_raw_sample,
        '占比(%)': 100.00,
        '备注': '去重后的原始全量样本'
    },
    {
        '统计指标': '独生子女样本量',
        '样本数量': only_child_sample,
        '占比(%)': round(only_child_sample / total_raw_sample * 100, 2) if total_raw_sample > 0 else 0,
        '备注': '原始样本中的独生子女，已从分析样本中剔除'
    },
    {
        '统计指标': '二孩家庭总样本量',
        '样本数量': two_child_total,
        '占比(%)': round(two_child_total / total_valid_sample * 100, 2) if total_valid_sample > 0 else 0,
        '备注': '有效分析样本中的二孩家庭'
    },
    {
        '统计指标': '二孩家庭中有姐姐的样本量',
        '样本数量': two_child_has_sister,
        '占比(%)': round(two_child_sister_ratio, 2),
        '备注': '占二孩家庭总样本的比例'
    },
    {
        '统计指标': '多孩家庭总样本量',
        '样本数量': multi_child_total,
        '占比(%)': round(multi_child_total / total_valid_sample * 100, 2) if total_valid_sample > 0 else 0,
        '备注': '有效分析样本中的多孩家庭（3个及以上孩子）'
    },
    {
        '统计指标': '多孩家庭中有姐姐的样本量',
        '样本数量': multi_child_has_sister,
        '占比(%)': round(multi_child_sister_ratio, 2),
        '备注': '占多孩家庭总样本的比例'
    }
])

# 导出Excel文件
sample_stats_df.to_excel(r'D:\w\T统计学\期末论文\期末数据\新表\家庭结构与姐姐样本统计.xlsx', index=False)
print("✅ 已生成：家庭结构与姐姐样本统计.xlsx")

# 打印样本构成统计（原有逻辑）
print("=" * 70)
print("样本构成统计")
print("=" * 70)
print(f"总有效样本: {len(data)}")
print(f"二孩_只有姐姐: {(data['two_child_only_sister'] == 1).sum()}")
print(f"多孩_有姐姐及其他: {(data['multi_with_sister'] == 1).sum()}")
print(f"无姐姐对照组: {(data['no_sister'] == 1).sum()}")

# ====================== 10. 分样本回归（修正版：二孩和多孩样本不重叠） ======================
print("\n" + "=" * 70)
print("【方法一：分样本回归】")
print("=" * 70)

# 定义要分析的被解释变量
outcome_vars = {
    'std_chn': '语文',
    'std_mat': '数学',
    'std_eng': '英语',
    'gpa': '综合(GPA)'
}

# 存储分样本回归结果
results_split = {
    '二孩家庭': {},
    '多孩家庭': {}
}

for var, label in outcome_vars.items():
    # ===== 二孩家庭回归 =====
    # 处理组：二孩家庭中只有姐姐的
    df_two_treatment = data[data['two_child_only_sister'] == 1].copy()
    # 对照组：二孩家庭中无姐姐的（只有哥哥、或只有弟弟妹妹）
    df_two_control = data[(data['total_siblings'] == 1) & (data['has_sister'] == 0)].copy()
    df_two = pd.concat([df_two_treatment, df_two_control], ignore_index=True)
    df_two['treatment'] = df_two['two_child_only_sister'].astype(int)
    df_two = df_two.dropna(subset=[var])

    if len(df_two) > 0:
        X_two = sm.add_constant(df_two[['treatment', 'gender', 'par_edu', 'hukou']])
        model_two = sm.OLS(df_two[var], X_two).fit(cov_type='HC3')

        results_split['二孩家庭'][label] = {
            'coef': model_two.params['treatment'],
            'se': model_two.bse['treatment'],
            'p': model_two.pvalues['treatment'],
            'n_treatment': len(df_two_treatment),
            'n_control': len(df_two_control),
            'n_total': len(df_two),
            'r2': model_two.rsquared
        }

        # 保存到全局变量供后续使用（使用综合GPA的结果）
        if var == 'gpa':
            df_two_final = df_two.copy()
            X_two_final = X_two.copy()
            model_two_final = model_two

    # ===== 多孩家庭回归 =====
    # 处理组：多孩家庭中有姐姐的
    df_multi_treatment = data[(data['total_siblings'] >= 2) & (data['has_sister'] == 1)].copy()
    # 对照组：多孩家庭中无姐姐的
    df_multi_control = data[(data['total_siblings'] >= 2) & (data['has_sister'] == 0)].copy()
    df_multi = pd.concat([df_multi_treatment, df_multi_control], ignore_index=True)
    df_multi['treatment'] = df_multi['has_sister'].astype(int)
    df_multi = df_multi.dropna(subset=[var])

    if len(df_multi) > 0:
        X_multi = sm.add_constant(df_multi[['treatment', 'gender', 'par_edu', 'hukou', 'total_siblings']])
        model_multi = sm.OLS(df_multi[var], X_multi).fit(cov_type='HC3')

        results_split['多孩家庭'][label] = {
            'coef': model_multi.params['treatment'],
            'se': model_multi.bse['treatment'],
            'p': model_multi.pvalues['treatment'],
            'n_treatment': len(df_multi_treatment),
            'n_control': len(df_multi_control),
            'n_total': len(df_multi),
            'r2': model_multi.rsquared
        }

        # 保存到全局变量供后续使用（使用综合GPA的结果）
        if var == 'gpa':
            df_multi_final = df_multi.copy()
            X_multi_final = X_multi.copy()
            model_multi_final = model_multi

    # 打印结果
    print(f"\n【{label}】")
    print(
        f"  二孩家庭: β={results_split['二孩家庭'][label]['coef']:.4f}, p={results_split['二孩家庭'][label]['p']:.4f}, N={results_split['二孩家庭'][label]['n_total']}")
    print(
        f"  多孩家庭: β={results_split['多孩家庭'][label]['coef']:.4f}, p={results_split['多孩家庭'][label]['p']:.4f}, N={results_split['多孩家庭'][label]['n_total']}")

# ====================== 覆盖原始变量，供后续第13、15节使用 ======================
df_two = df_two_final.copy()
X_two = X_two_final.copy()
model_two = model_two_final

df_multi = df_multi_final.copy()
X_multi = X_multi_final.copy()
model_multi = model_multi_final

# ====================== 分样本回归汇总表（论文格式） ======================
print("\n" + "=" * 70)
print("【分样本回归结果汇总表】")
print("=" * 70)

subject_columns = list(outcome_vars.values())

# 构建表格数据
table_data_split = []

# 二孩家庭行
row_two = {'模型': '二孩家庭'}
for subject in subject_columns:
    res = results_split['二孩家庭'][subject]
    coef = res['coef']
    se = res['se']
    p = res['p']

    if p < 0.01:
        stars = '***'
    elif p < 0.05:
        stars = '**'
    elif p < 0.1:
        stars = '*'
    else:
        stars = ''

    row_two[subject] = f"{coef:.4f}{stars}\n({se:.4f})"
table_data_split.append(row_two)

# 多孩家庭行
row_multi = {'模型': '多孩家庭'}
for subject in subject_columns:
    res = results_split['多孩家庭'][subject]
    coef = res['coef']
    se = res['se']
    p = res['p']

    if p < 0.01:
        stars = '***'
    elif p < 0.05:
        stars = '**'
    elif p < 0.1:
        stars = '*'
    else:
        stars = ''

    row_multi[subject] = f"{coef:.4f}{stars}\n({se:.4f})"
table_data_split.append(row_multi)

# 样本量行（处理组/对照组/总计）- 使用综合GPA的样本量统一显示
for family_type in ['二孩家庭', '多孩家庭']:
    res_gpa = results_split[family_type]['综合(GPA)']
    row_n = {'模型': f'{family_type}样本量'}
    for subject in subject_columns:
        row_n[subject] = f"{res_gpa['n_treatment']}"
    table_data_split.append(row_n)

# R²行
for family_type in ['二孩家庭', '多孩家庭']:
    row_r2 = {'模型': f'{family_type} R²'}
    for subject in subject_columns:
        row_r2[subject] = f"{results_split[family_type][subject]['r2']:.4f}"
    table_data_split.append(row_r2)

# 转换为DataFrame
df_split_results = pd.DataFrame(table_data_split)
df_split_results = df_split_results[['模型'] + subject_columns]

# 导出Excel
df_split_results.to_excel(r'D:\w\T统计学\期末论文\期末数据\新表\分样本回归结果_修正版.xlsx', index=False)
print("\n✅ 分样本回归结果已保存: 分样本回归结果_修正版.xlsx")

# 打印表格预览
print("\n表格预览：")
print("=" * 100)
print(f"{'模型':22}", end='')
for subject in subject_columns:
    print(f" | {subject:28}", end='')
print()
print("-" * 100)
for _, row in df_split_results.iterrows():
    print(f"{row['模型']:22}", end='')
    for subject in subject_columns:
        cell_str = str(row[subject]).replace('\n', ' ')
        print(f" | {cell_str[:28]:28}", end='')
    print()
print("=" * 100)
# 验证样本覆盖情况
print("\n" + "=" * 70)
print("【样本覆盖验证】")
print("=" * 70)

# 二孩家庭总数
two_total = (data['total_siblings'] == 1).sum()
print(f"二孩家庭总数: {two_total}")

# 多孩家庭总数
multi_total = (data['total_siblings'] >= 2).sum()
print(f"多孩家庭总数: {multi_total}")

# 总计
print(f"总计: {two_total + multi_total}")
print(f"总有效样本: {len(data)}")

# 检查是否有 gap
if two_total + multi_total != len(data):
    print(f"⚠️ 存在差异: {len(data) - (two_total + multi_total)} 个样本未被归类")

# 检查二孩家庭内部
two_has_sister = (data['total_siblings'] == 1) & (data['has_sister'] == 1)
two_no_sister = (data['total_siblings'] == 1) & (data['has_sister'] == 0)
print(f"\n二孩有姐姐: {two_has_sister.sum()}")
print(f"二孩无姐姐: {two_no_sister.sum()}")
print(f"二孩合计: {two_has_sister.sum() + two_no_sister.sum()}")

# 检查多孩家庭内部
multi_has_sister = (data['total_siblings'] >= 2) & (data['has_sister'] == 1)
multi_no_sister = (data['total_siblings'] >= 2) & (data['has_sister'] == 0)
print(f"\n多孩有姐姐: {multi_has_sister.sum()}")
print(f"多孩无姐姐: {multi_no_sister.sum()}")
print(f"多孩合计: {multi_has_sister.sum() + multi_no_sister.sum()}")

# 检查 two_child_only_sister 定义
two_child_only = (data['two_child_only_sister'] == 1).sum()
print(f"\ntwo_child_only_sister (只有姐姐的二孩): {two_child_only}")

# 对比：二孩有姐姐 vs 只有姐姐的二孩
print(f"二孩有姐姐总数: {two_has_sister.sum()}")
print(f"其中只有姐姐(无哥哥弟弟): {two_child_only}")
print(f"二孩有姐姐但有哥哥或弟弟: {two_has_sister.sum() - two_child_only}")
# ====================== 11. 交互项检验（原有逻辑） ======================
print("\n" + "=" * 70)
print("【方法二：交互项检验】")
print("=" * 70)

data_inter = data[data['has_sister'].isin([0, 1])].copy()
X_inter = sm.add_constant(data_inter[['has_sister', 'is_multi', 'sister_multi', 'gender', 'par_edu', 'hukou']])
model_inter = sm.OLS(data_inter['gpa'], X_inter).fit(cov_type='HC3')
# ⭐ 添加这一行：保存原始的交互项模型，避免被后续覆盖
model_inter_original = model_inter
data_inter_original = data_inter
print(f"交互项(姐姐×多孩)系数: β={model_inter.params['sister_multi']:.4f}, p={model_inter.pvalues['sister_multi']:.4f}")
# ====================== 11. 交互项检验（分学科） ======================
print("\n" + "=" * 70)
print("【交互项检验：分学科结果】")
print("=" * 70)

# 定义要分析的被解释变量
outcome_vars = {
    'std_chn': '语文',
    'std_mat': '数学',
    'std_eng': '英语',
    'gpa': '综合(GPA)'
}

# 存储各学科模型结果
subject_models = {}

for var, label in outcome_vars.items():
    # 清理缺失值
    data_subject = data[data['has_sister'].isin([0, 1])].dropna(subset=[var]).copy()

    # 构建回归模型
    X = sm.add_constant(data_subject[['has_sister', 'is_multi', 'sister_multi', 'gender', 'par_edu', 'hukou']])
    model = sm.OLS(data_subject[var], X).fit(cov_type='HC3')
    subject_models[label] = model

    print(f"\n【{label}】")
    print(f"有姐姐系数: β={model.params['has_sister']:.4f}, p={model.pvalues['has_sister']:.4f}")
    print(f"交互项(姐姐×多孩)系数: β={model.params['sister_multi']:.4f}, p={model.pvalues['sister_multi']:.4f}")
    print(f"样本量: {len(data_subject)}, R²: {model.rsquared:.4f}")

# ====================== 分学科交互项汇总表（论文格式） ======================
print("\n" + "=" * 70)
print("【分学科交互项回归结果汇总表】")
print("=" * 70)

# 定义需要提取的变量
var_list = ['has_sister', 'sister_multi', 'gender', 'par_edu', 'hukou']
var_labels = {
    'has_sister': '有姐姐',
    'sister_multi': '有姐姐 × 多孩家庭',
    'gender': '学生性别',
    'par_edu': '父母平均教育年限',
    'hukou': '城镇户口'
}

# 构建表格数据
table_data = []
for var in var_list:
    row = {'变量': var_labels[var]}
    for subject_label, model in subject_models.items():
        if var in model.params:
            coef = model.params[var]
            se = model.bse[var]
            p = model.pvalues[var]

            # 显著性星号
            if p < 0.01:
                stars = '***'
            elif p < 0.05:
                stars = '**'
            elif p < 0.1:
                stars = '*'
            else:
                stars = ''

            # 格式化：系数*** (标准误)
            row[subject_label] = f"{coef:.4f}{stars}\n({se:.4f})"
        else:
            row[subject_label] = '—'
    table_data.append(row)

# 添加样本量和R²行
row_n = {'变量': '样本量'}
row_r2 = {'变量': 'R²'}
for subject_label, model in subject_models.items():
    row_n[subject_label] = model.nobs
    row_r2[subject_label] = f"{model.rsquared:.4f}"
table_data.append(row_n)
table_data.append(row_r2)

# 转换为DataFrame并导出
df_subject_results = pd.DataFrame(table_data)
subject_columns = list(subject_models.keys())
df_subject_results = df_subject_results[['变量'] + subject_columns]

# 导出Excel
df_subject_results.to_excel(r'D:\w\T统计学\期末论文\期末数据\新表\分学科交互项回归结果.xlsx', index=False)
print("\n✅ 分学科交互项回归结果已保存: 分学科交互项回归结果.xlsx")

# 打印表格预览
print("\n表格预览：")
print("=" * 100)
for _, row in df_subject_results.iterrows():
    print(f"{row['变量']:15}", end='')
    for subject in subject_columns:
        cell_str = str(row[subject]).replace('\n', ' ')
        print(f" | {cell_str[:28]:28}", end='')
    print()
print("=" * 100)
# ====================== 12. 异质性分析（原有逻辑） ======================
print("\n" + "=" * 70)
print("【方法三：异质性分析】")
print("=" * 70)

results_hetero = {}

# 11.1 分性别（只在二孩家庭中）
df_two_hetero = data[data['two_child_only_sister'].isin([1, 0]) & (data['no_sister'].isin([1, 0]))].copy()
df_two_hetero = df_two_hetero[(df_two_hetero['two_child_only_sister'] == 1) | (df_two_hetero['no_sister'] == 1)]
df_two_hetero['treatment'] = df_two_hetero['two_child_only_sister'].astype(int)

for gender_val, gender_name in [(1, '男生(弟弟)'), (0, '女生(妹妹)')]:
    sub = df_two_hetero[df_two_hetero['gender'] == gender_val]
    if len(sub) > 10:
        X_sub = sm.add_constant(sub[['treatment', 'par_edu', 'hukou']])
        model_sub = sm.OLS(sub['gpa'], X_sub).fit(cov_type='HC3')
        results_hetero[f'分性别_{gender_name}'] = {
            '系数': model_sub.params['treatment'],
            'p值': model_sub.pvalues['treatment'],
            '样本量': len(sub)
        }
        print(
            f"{gender_name}: β={model_sub.params['treatment']:.4f}, p={model_sub.pvalues['treatment']:.4f}, N={len(sub)}")

# 11.2 分学科（只在二孩家庭中）
for subj, name in zip(['std_chn', 'std_mat', 'std_eng'], ['语文', '数学', '英语']):
    sub_data = df_two_hetero[[subj, 'treatment', 'par_edu', 'hukou']].dropna()
    if len(sub_data) > 10:
        X_sub = sm.add_constant(sub_data[['treatment', 'par_edu', 'hukou']])
        model_sub = sm.OLS(sub_data[subj], X_sub).fit(cov_type='HC3')
        results_hetero[f'分学科_{name}'] = {
            '系数': model_sub.params['treatment'],
            'p值': model_sub.pvalues['treatment'],
            '样本量': len(sub_data)
        }
        print(
            f"{name}: β={model_sub.params['treatment']:.4f}, p={model_sub.pvalues['treatment']:.4f}, N={len(sub_data)}")

# 11.3 分户籍（只在二孩家庭中）
for hukou_val, hukou_name in [(1, '城镇户口'), (0, '农村户口')]:
    sub = df_two_hetero[df_two_hetero['hukou'] == hukou_val]
    if len(sub) > 10:
        X_sub = sm.add_constant(sub[['treatment', 'gender', 'par_edu']])
        model_sub = sm.OLS(sub['gpa'], X_sub).fit(cov_type='HC3')
        results_hetero[f'分户籍_{hukou_name}'] = {
            '系数': model_sub.params['treatment'],
            'p值': model_sub.pvalues['treatment'],
            '样本量': len(sub)
        }
        print(
            f"{hukou_name}: β={model_sub.params['treatment']:.4f}, p={model_sub.pvalues['treatment']:.4f}, N={len(sub)}")

# 11.4 分父母教育水平（中位数分组）
median_edu = data['par_edu'].median()
for edu_val, edu_name in [(1, '父母教育_高'), (0, '父母教育_低')]:
    if edu_val == 1:
        sub = df_two_hetero[df_two_hetero['par_edu'] >= median_edu]
    else:
        sub = df_two_hetero[df_two_hetero['par_edu'] < median_edu]
    if len(sub) > 10:
        X_sub = sm.add_constant(sub[['treatment', 'gender', 'hukou']])
        model_sub = sm.OLS(sub['gpa'], X_sub).fit(cov_type='HC3')
        results_hetero[f'分父母教育_{edu_name}'] = {
            '系数': model_sub.params['treatment'],
            'p值': model_sub.pvalues['treatment'],
            '样本量': len(sub)
        }
        print(
            f"{edu_name}: β={model_sub.params['treatment']:.4f}, p={model_sub.pvalues['treatment']:.4f}, N={len(sub)}")

# ====================== 13. 稳健性检验（原有逻辑） ======================
print("\n" + "=" * 70)
print("【方法四：稳健性检验】")
print("=" * 70)

results_robust = {}

# 稳健性1：剔除成绩极端值（前后1%）
data_robust1 = data[(data['gpa'] >= data['gpa'].quantile(0.01)) & (data['gpa'] <= data['gpa'].quantile(0.99))].copy()
df_two_r1 = data_robust1[(data_robust1['two_child_only_sister'] == 1) | (data_robust1['no_sister'] == 1)]
df_two_r1['treatment'] = df_two_r1['two_child_only_sister'].astype(int)
X_r1 = sm.add_constant(df_two_r1[['treatment', 'gender', 'par_edu', 'hukou']])
model_r1 = sm.OLS(df_two_r1['gpa'], X_r1).fit(cov_type='HC3')
results_robust['剔除极端值_二孩'] = {'系数': model_r1.params['treatment'], 'p值': model_r1.pvalues['treatment']}
print(f"剔除极端值（二孩家庭）: β={model_r1.params['treatment']:.4f}, p={model_r1.pvalues['treatment']:.4f}")

df_multi_r1 = data_robust1[(data_robust1['multi_with_sister'] == 1) | (data_robust1['no_sister'] == 1)]
df_multi_r1['treatment'] = df_multi_r1['multi_with_sister'].astype(int)
X_multi_r1 = sm.add_constant(df_multi_r1[['treatment', 'gender', 'par_edu', 'hukou', 'total_siblings']])
model_multi_r1 = sm.OLS(df_multi_r1['gpa'], X_multi_r1).fit(cov_type='HC3')
results_robust['剔除极端值_多孩'] = {'系数': model_multi_r1.params['treatment'],
                                     'p值': model_multi_r1.pvalues['treatment']}
print(f"剔除极端值（多孩家庭）: β={model_multi_r1.params['treatment']:.4f}, p={model_multi_r1.pvalues['treatment']:.4f}")

# 稳健性2：更换GPA定义（等权重原始分数，不标准化）
data['gpa_raw'] = data[['tr_chn', 'tr_mat', 'tr_eng']].mean(axis=1)
df_two_r2 = data[(data['two_child_only_sister'] == 1) | (data['no_sister'] == 1)].copy()
df_two_r2['treatment'] = df_two_r2['two_child_only_sister'].astype(int)
X_r2 = sm.add_constant(df_two_r2[['treatment', 'gender', 'par_edu', 'hukou']])
model_r2 = sm.OLS(df_two_r2['gpa_raw'], X_r2).fit(cov_type='HC3')
results_robust['原始分数GPA_二孩'] = {'系数': model_r2.params['treatment'], 'p值': model_r2.pvalues['treatment']}
print(f"原始分数GPA（二孩家庭）: β={model_r2.params['treatment']:.4f}, p={model_r2.pvalues['treatment']:.4f}")

# 稳健性3：更换标准误聚类方式（普通标准误）
model_two_cluster = sm.OLS(df_two['gpa'], X_two).fit()
results_robust['标准误_普通_二孩'] = {'系数': model_two_cluster.params['treatment'],
                                      'p值': model_two_cluster.pvalues['treatment']}
print(
    f"普通标准误（二孩家庭）: β={model_two_cluster.params['treatment']:.4f}, p={model_two_cluster.pvalues['treatment']:.4f}")

# 稳健性4：Bootstrap标准误（修复版）
np.random.seed(42)
bootstrap_betas = []
n_bootstrap = 500
df_two_boot = df_two.copy()

for _ in range(n_bootstrap):
    boot_idx = np.random.choice(len(df_two_boot), len(df_two_boot), replace=True)
    boot_data = df_two_boot.iloc[boot_idx]
    X_boot = sm.add_constant(boot_data[['treatment', 'gender', 'par_edu', 'hukou']])
    try:
        boot_model = sm.OLS(boot_data['gpa'], X_boot).fit()
        bootstrap_betas.append(boot_model.params['treatment'])
    except:
        pass

bootstrap_betas = np.array(bootstrap_betas)
bootstrap_mean = np.mean(bootstrap_betas)
bootstrap_se = np.std(bootstrap_betas, ddof=1)

# 修复：使用正态分布计算p值
if bootstrap_se > 0:
    z_score = bootstrap_mean / bootstrap_se
    bootstrap_p = 2 * (1 - norm.cdf(abs(z_score)))
else:
    bootstrap_p = 1.0

# 同时计算百分位数置信区间
ci_lower = np.percentile(bootstrap_betas, 2.5)
ci_upper = np.percentile(bootstrap_betas, 97.5)

results_robust['Bootstrap标准误_二孩'] = {
    '系数': bootstrap_mean,
    'p值': bootstrap_p,
    'se': bootstrap_se,
    'ci_lower': ci_lower,
    'ci_upper': ci_upper
}
print(f"Bootstrap（二孩家庭）: β={bootstrap_mean:.4f}, p={bootstrap_p:.4f}, SE={bootstrap_se:.4f}")
print(f"  95% Bootstrap CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

# 稳健性5：加入更多控制变量（家庭经济）
if 'stfeco' in data.columns:
    data['family_econ'] = data['stfeco']
    df_two_r4 = data[(data['two_child_only_sister'] == 1) | (data['no_sister'] == 1)].copy()
    df_two_r4['treatment'] = df_two_r4['two_child_only_sister'].astype(int)
    X_r4 = sm.add_constant(df_two_r4[['treatment', 'gender', 'par_edu', 'hukou', 'family_econ']])
    model_r4 = sm.OLS(df_two_r4['gpa'], X_r4).fit(cov_type='HC3')
    results_robust['加家庭经济_二孩'] = {'系数': model_r4.params['treatment'], 'p值': model_r4.pvalues['treatment']}
    print(f"加入家庭经济（二孩家庭）: β={model_r4.params['treatment']:.4f}, p={model_r4.pvalues['treatment']:.4f}")
# ====================== 稳健性检验汇总表（论文格式：变量为行，方法为列） ======================
print("\n" + "=" * 70)
print("【稳健性检验汇总表（论文格式）】")
print("=" * 70)

# 存储各模型的结果
robust_models = {}

# 模型1：剔除极端值（二孩家庭交互项模型）
# 需要重新跑一个剔除极端值后的交互项模型
data_robust1 = data[(data['gpa'] >= data['gpa'].quantile(0.01)) & (data['gpa'] <= data['gpa'].quantile(0.99))].copy()
data_inter_r1 = data_robust1[data_robust1['has_sister'].isin([0, 1])].copy()
X_inter_r1 = sm.add_constant(data_inter_r1[['has_sister', 'is_multi', 'sister_multi', 'gender', 'par_edu', 'hukou']])
model_inter_r1 = sm.OLS(data_inter_r1['gpa'], X_inter_r1).fit(cov_type='HC3')
robust_models['剔除极端值'] = {
    'model': model_inter_r1,
    'n': len(data_inter_r1),
    'r2': model_inter_r1.rsquared
}

# 模型2：更换被解释变量（原始分数GPA，交互项模型）
data['gpa_raw'] = data[['tr_chn', 'tr_mat', 'tr_eng']].mean(axis=1)
data_inter_r2 = data[data['has_sister'].isin([0, 1])].copy()
X_inter_r2 = sm.add_constant(data_inter_r2[['has_sister', 'is_multi', 'sister_multi', 'gender', 'par_edu', 'hukou']])
model_inter_r2 = sm.OLS(data_inter_r2['gpa_raw'], X_inter_r2).fit(cov_type='HC3')
robust_models['更换被解释变量'] = {
    'model': model_inter_r2,
    'n': len(data_inter_r2),
    'r2': model_inter_r2.rsquared
}

# 模型3：普通标准误（不聚类）
data_inter_r3 = data[data['has_sister'].isin([0, 1])].copy()
X_inter_r3 = sm.add_constant(data_inter_r3[['has_sister', 'is_multi', 'sister_multi', 'gender', 'par_edu', 'hukou']])
model_inter_r3 = sm.OLS(data_inter_r3['gpa'], X_inter_r3).fit()  # 普通标准误
robust_models['普通标准误'] = {
    'model': model_inter_r3,
    'n': len(data_inter_r3),
    'r2': model_inter_r3.rsquared
}

# 模型4：Bootstrap标准误（使用之前计算的bootstrap结果，但这里需要交互项模型的bootstrap）
# 重新计算交互项模型的Bootstrap标准误
np.random.seed(42)
bootstrap_betas_inter = {'has_sister': [], 'sister_multi': []}
n_bootstrap = 500
data_inter_boot = data[data['has_sister'].isin([0, 1])].copy()

for _ in range(n_bootstrap):
    boot_idx = np.random.choice(len(data_inter_boot), len(data_inter_boot), replace=True)
    boot_data = data_inter_boot.iloc[boot_idx]
    X_boot = sm.add_constant(boot_data[['has_sister', 'is_multi', 'sister_multi', 'gender', 'par_edu', 'hukou']])
    try:
        boot_model = sm.OLS(boot_data['gpa'], X_boot).fit()
        bootstrap_betas_inter['has_sister'].append(boot_model.params['has_sister'])
        bootstrap_betas_inter['sister_multi'].append(boot_model.params['sister_multi'])
    except:
        pass

# 获取原始模型的系数
model_inter_original = robust_models['剔除极端值']['model']  # 用剔除极端值的系数作为基准
original_coef_has_sister = model_inter_original.params['has_sister']
original_coef_sister_multi = model_inter_original.params['sister_multi']

# 计算Bootstrap标准误和p值
bootstrap_se_has_sister = np.std(bootstrap_betas_inter['has_sister'], ddof=1)
bootstrap_se_sister_multi = np.std(bootstrap_betas_inter['sister_multi'], ddof=1)
bootstrap_p_has_sister = 2 * (
            1 - norm.cdf(abs(original_coef_has_sister / bootstrap_se_has_sister))) if bootstrap_se_has_sister > 0 else 1
bootstrap_p_sister_multi = 2 * (1 - norm.cdf(
    abs(original_coef_sister_multi / bootstrap_se_sister_multi))) if bootstrap_se_sister_multi > 0 else 1


# 构造一个虚拟模型对象存储Bootstrap结果
class BootstrapModel:
    def __init__(self, params, bse, pvalues, rsquared, n):
        self.params = params
        self.bse = bse
        self.pvalues = pvalues
        self.rsquared = rsquared
        self.n = n


bootstrap_model = BootstrapModel(
    params={'has_sister': original_coef_has_sister, 'sister_multi': original_coef_sister_multi},
    bse={'has_sister': bootstrap_se_has_sister, 'sister_multi': bootstrap_se_sister_multi},
    pvalues={'has_sister': bootstrap_p_has_sister, 'sister_multi': bootstrap_p_sister_multi},
    rsquared=model_inter_original.rsquared,
    n=len(data_inter_boot)
)
robust_models['Bootstrap标准误'] = {
    'model': bootstrap_model,
    'n': len(data_inter_boot),
    'r2': model_inter_original.rsquared
}

# 模型5：加入家庭经济控制变量
if 'stfeco' in data.columns:
    data['family_econ'] = data['stfeco']
    data_inter_r5 = data[data['has_sister'].isin([0, 1])].dropna(subset=['family_econ']).copy()
    X_inter_r5 = sm.add_constant(
        data_inter_r5[['has_sister', 'is_multi', 'sister_multi', 'gender', 'par_edu', 'hukou', 'family_econ']])
    model_inter_r5 = sm.OLS(data_inter_r5['gpa'], X_inter_r5).fit(cov_type='HC3')
    robust_models['加入家庭经济'] = {
        'model': model_inter_r5,
        'n': len(data_inter_r5),
        'r2': model_inter_r5.rsquared
    }

# ====================== 构建汇总表格 ======================
# 定义需要提取的变量
variables = ['has_sister', 'sister_multi', 'gender', 'par_edu', 'hukou']
var_labels = {
    'has_sister': '有姐姐',
    'sister_multi': '有姐姐×多孩家庭',
    'gender': '学生性别',
    'par_edu': '父母平均教育年限',
    'hukou': '城镇户口'
}

# 构建表格数据
table_data = []
for var in variables:
    row = {'变量': var_labels[var]}
    for method_name, method_info in robust_models.items():
        model = method_info['model']
        if var in model.params:
            coef = model.params[var]
            se = model.bse[var]
            p = model.pvalues[var]

            # 显著性星号
            if p < 0.01:
                stars = '***'
            elif p < 0.05:
                stars = '**'
            elif p < 0.1:
                stars = '*'
            else:
                stars = ''

            # 格式化单元格内容：系数*** (标准误)
            cell = f"{coef:.4f}{stars}\n({se:.4f})"
            row[method_name] = cell
        else:
            row[method_name] = '—'
    table_data.append(row)

# 添加样本量和R²行
row_n = {'变量': '样本量'}
row_r2 = {'变量': 'R²'}
for method_name, method_info in robust_models.items():
    row_n[method_name] = method_info['n']
    row_r2[method_name] = f"{method_info['r2']:.4f}"
table_data.append(row_n)
table_data.append(row_r2)

# 转换为DataFrame
df_robust_table = pd.DataFrame(table_data)

# 重命名列
method_columns = list(robust_models.keys())
df_robust_table = df_robust_table[['变量'] + method_columns]

# 导出Excel
df_robust_table.to_excel(r'D:\w\T统计学\期末论文\期末数据\新表\稳健性检验汇总表_论文格式.xlsx', index=False)
print("\n✅ 稳健性检验汇总表（论文格式）已保存: 稳健性检验汇总表_论文格式.xlsx")

# 打印表格预览
print("\n表格预览：")
print("=" * 100)
for _, row in df_robust_table.iterrows():
    print(f"{row['变量']:15}", end='')
    for method in method_columns:
        print(f" | {str(row[method])[:20]:20}", end='')
    print()
print("=" * 100)
# ====================== 14. 竞争效应补充分析（原有逻辑） ======================
print("\n" + "=" * 70)
print("【补充分析】多孩家庭内部：有哥哥 vs 无哥哥")
print("=" * 70)

multi_sister = data[data['multi_with_sister'] == 1].copy()
if len(multi_sister) > 0:
    multi_sister['has_brother'] = (multi_sister['b0201'] >= 1).astype(int)
    X_bro = sm.add_constant(multi_sister[['has_brother', 'gender', 'par_edu', 'hukou', 'total_siblings']])
    model_bro = sm.OLS(multi_sister['gpa'], X_bro).fit(cov_type='HC3')
    print(f"有哥哥（vs 无哥哥）效应: β={model_bro.params['has_brother']:.4f}, p={model_bro.pvalues['has_brother']:.4f}")

    # 描述性统计
    has_bro = multi_sister[multi_sister['has_brother'] == 1]
    no_bro = multi_sister[multi_sister['has_brother'] == 0]
    print(f"  有哥哥组 GPA均值: {has_bro['gpa'].mean():.4f} (N={len(has_bro)})")
    print(f"  无哥哥组 GPA均值: {no_bro['gpa'].mean():.4f} (N={len(no_bro)})")

# ====================== 15. 汇总结果并导出Excel（原有逻辑） ======================
print("\n" + "=" * 70)
print("【结果导出】")
print("=" * 70)

# 创建汇总表格
summary_results = []

# 分样本回归
summary_results.append({
    '方法': '分样本回归',
    '子组': '二孩家庭（只有1个姐姐 vs 无姐姐）',
    '系数': model_two.params['treatment'],
    'p值': model_two.pvalues['treatment'],
    '标准误': model_two.bse['treatment'],
    '样本量': len(df_two)
})
summary_results.append({
    '方法': '分样本回归',
    '子组': '多孩家庭（有姐姐+其他弟妹 vs 无姐姐）',
    '系数': model_multi.params['treatment'],
    'p值': model_multi.pvalues['treatment'],
    '标准误': model_multi.bse['treatment'],
    '样本量': len(df_multi)
})

# 交互项检验
summary_results.append({
    '方法': '交互项检验',
    '子组': '姐姐 × 多孩',
    '系数': model_inter_original.params['sister_multi'],
    'p值': model_inter_original.pvalues['sister_multi'],
    '标准误': model_inter_original.bse['sister_multi'],
    '样本量': len(data_inter_original)
})
# 异质性分析
for key, val in results_hetero.items():
    summary_results.append({
        '方法': '异质性分析',
        '子组': key,
        '系数': val['系数'],
        'p值': val['p值'],
        '标准误': None,
        '样本量': val['样本量']
    })

# 稳健性检验
for key, val in results_robust.items():
    summary_results.append({
        '方法': '稳健性检验',
        '子组': key,
        '系数': val['系数'],
        'p值': val['p值'],
        '标准误': val.get('se', None),
        '样本量': None
    })

# 补充分析
if len(multi_sister) > 0:
    summary_results.append({
        '方法': '补充分析',
        '子组': '多孩有姐姐家庭_有哥哥效应',
        '系数': model_bro.params['has_brother'],
        'p值': model_bro.pvalues['has_brother'],
        '标准误': model_bro.bse['has_brother'],
        '样本量': len(multi_sister)
    })

# 转换为DataFrame并导出
df_results = pd.DataFrame(summary_results)
df_results.to_excel(r'D:\w\T统计学\期末论文\期末数据\新表\完整分析结果_二孩vs多孩.xlsx', index=False)

print("\n✅ 结果已保存: 完整分析结果_二孩vs多孩.xlsx")

# ====================== 16. 描述性统计表（原有逻辑） ======================
desc_stats = []
for group_name in ['二孩_只有姐姐', '多孩_有姐姐及其他', '无姐姐']:
    if group_name == '二孩_只有姐姐':
        group_data = data[data['two_child_only_sister'] == 1]
    elif group_name == '多孩_有姐姐及其他':
        group_data = data[data['multi_with_sister'] == 1]
    else:
        group_data = data[data['no_sister'] == 1]

    desc_stats.append({
        '组别': group_name,
        '样本量(N)': len(group_data),
        'GPA均值': group_data['gpa'].mean(),
        'GPA标准差': group_data['gpa'].std(),
        '语文均值': group_data['std_chn'].mean() if 'std_chn' in group_data.columns else None,
        '数学均值': group_data['std_mat'].mean() if 'std_mat' in group_data.columns else None,
        '英语均值': group_data['std_eng'].mean() if 'std_eng' in group_data.columns else None,
        '父母教育均值': group_data['par_edu'].mean(),
        '城镇比例': group_data['hukou'].mean()
    })

df_desc = pd.DataFrame(desc_stats)
df_desc.to_excel(r'D:\w\T统计学\期末论文\期末数据\新表\描述性统计_分组对比.xlsx', index=False)
print("\n✅ 描述性统计已保存: 描述性统计_分组对比.xlsx")

print("\n" + "=" * 70)
print("分析完成！共生成3个Excel文件：")
print("  1. 家庭结构与姐姐样本统计.xlsx（你新增的需求）")
print("  2. 完整分析结果_二孩vs多孩.xlsx")
print("  3. 描述性统计_分组对比.xlsx")
print("=" * 70)

# 检查同胞数量的分布
print(data['total_siblings'].value_counts().sort_index())

# 看看除了1和≥2之外，还有哪些值
others = data[~data['total_siblings'].isin([1,2,3,4])]
print(f"\n异常值数量: {len(others)}")
print(others['total_siblings'].unique())

# 检查total_siblings=0的样本
zero_sib = data[data['total_siblings'] == 0]
print(f"\ntotal_siblings=0 的样本量: {len(zero_sib)}")
if len(zero_sib) > 0:
    print(zero_sib[['b01', 'b0201', 'b0202', 'b0203', 'b0204']].head())