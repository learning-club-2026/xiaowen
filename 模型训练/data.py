# 数据处理
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from setting import RANDOM_SEED

def load_and_preprocess():
    # -------------------- 读取数据 --------------------
    train = pd.read_csv("house-prices-advanced-regression-techniques_副本/train.csv")
    test = pd.read_csv("house-prices-advanced-regression-techniques_副本/test.csv")

    X_train_raw = train.drop(["SalePrice", "Id"], axis=1)   
    y_train_raw = train["SalePrice"]
    X_test_raw = test.drop(["Id"], axis=1)                  

    # 合并以便统一处理
    all_data = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

    # -------------------- 有序映射字典 --------------------
    quality_map = {
        'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1,
        'NA': 0, 'None': 0
    }
    finish_map = {
        'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0
    }
    exposure_map = {
        'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0
    }

    qual_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                 'HeatingQC', 'KitchenQual', 'FireplaceQu',
                 'GarageQual', 'GarageCond', 'PoolQC']
    finish_cols = ['BsmtFinType1', 'BsmtFinType2']
    exposure_cols = ['BsmtExposure']

    # -------------------- 缺失值填充 --------------------
    # LotFrontage 按邻居中位数填充
    all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median())
    )

    # GarageYrBlt：无车库填0，有车库缺失用YearBuilt
    all_data.loc[all_data['GarageType'].isna(), 'GarageYrBlt'] = 0
    all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna(all_data['YearBuilt'])

    # MasVnrArea：无饰面填0，其余缺失用中位数
    all_data.loc[all_data['MasVnrType'].isna() | (all_data['MasVnrType'] == 'None'), 'MasVnrArea'] = 0
    all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(all_data['MasVnrArea'].median())

    # Electrical 众数填充
    all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

    # 地下室数值列，缺失填0
    bsmt_num = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
    for col in bsmt_num:
        all_data[col] = all_data[col].fillna(0)

    # 车库数值列
    all_data['GarageCars'] = all_data['GarageCars'].fillna(0)
    all_data['GarageArea'] = all_data['GarageArea'].fillna(0)

    # 地下室浴室
    for col in ['BsmtFullBath', 'BsmtHalfBath']:
        all_data[col] = all_data[col].fillna(0)

    # -------------------- 映射有序特征 --------------------
    for col in qual_cols:
        all_data[col] = all_data[col].map(quality_map).fillna(0)
    for col in finish_cols:
        all_data[col] = all_data[col].map(finish_map).fillna(0)
    for col in exposure_cols:
        all_data[col] = all_data[col].map(exposure_map).fillna(0)

    # -------------------- 无序类别独热编码 --------------------
    categorical_cols = [
    'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',
    'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
    'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating', 'CentralAir',
    'Electrical', 'GarageType', 'GarageFinish', 'PavedDrive', 'Fence',
    'MiscFeature', 'SaleType', 'SaleCondition',
    'Functional'   
]
    all_data_encoded = pd.get_dummies(all_data, columns=categorical_cols, drop_first=True)

    # -------------------- 分离训练/测试 --------------------
    n_train = len(X_train_raw)
    X_train_enc = all_data_encoded.iloc[:n_train, :]
    X_test_enc = all_data_encoded.iloc[n_train:, :]

    # 目标变量取对数
    y_log = np.log(y_train_raw)

    # -------------------- 划分训练/验证（80/20） --------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_enc, y_log, test_size=0.2, random_state=RANDOM_SEED
    )

    # -------------------- 特征标准化 --------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test_enc)

    return {
        "X_train": X_train_scaled,
        "X_test": X_val_scaled,        
        "y_train": y_train.values,
        "y_test": y_val.values,
        "X_final_test": X_test_scaled  
    }