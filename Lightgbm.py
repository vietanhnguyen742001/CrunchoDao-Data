from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from scipy.stats import zscore
import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr
import lightgbm as lgb
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

top_features_dict = {
    'target_w': ['vratios_Feature_46', 'ekinetic_Feature_10', 'cmechanics_Feature_18', 'cmechanics_Feature_16', 'vratios_Feature_111', 'cmechanics_Feature_15', 'wrythm_Feature_146', 'cmechanics_Feature_17', 'wrythm_Feature_122', 'wrythm_Feature_98', 'wrythm_Feature_2', 'cmechanics_Feature_19', 'vratios_Feature_2', 'wrythm_Feature_50', 'vratios_Feature_47', 'wrythm_Feature_26', 'vratios_Feature_5', 'wrythm_Feature_170', 'cmechanics_Feature_14', 'vratios_Feature_3', 'ekinetic_Feature_23', 'wrythm_Feature_74', 'cmechanics_Feature_13', 'vratios_Feature_6', 'vratios_Feature_4', 'ekinetic_Feature_24', 'vratios_Feature_29', 'ekinetic_Feature_9', 'ekinetic_Feature_31', 'vratios_Feature_51'],
    'target_r': ['vratios_Feature_46', 'vratios_Feature_2', 'vratios_Feature_29', 'vratios_Feature_47', 'vratios_Feature_3', 'vratios_Feature_5', 'wrythm_Feature_185', 'vratios_Feature_83', 'vratios_Feature_4', 'vratios_Feature_84', 'vratios_Feature_95', 'vratios_Feature_39', 'vratios_Feature_42', 'vratios_Feature_96', 'vratios_Feature_6', 'wrythm_Feature_162', 'vratios_Feature_32', 'wrythm_Feature_66', 'vratios_Feature_51', 'wrythm_Feature_138', 'wrythm_Feature_42', 'vratios_Feature_63', 'wrythm_Feature_50', 'wrythm_Feature_186', 'vratios_Feature_94', 'wrythm_Feature_114', 'wrythm_Feature_90', 'vratios_Feature_45', 'wrythm_Feature_170', 'wrythm_Feature_146'],
    'target_g': ['vratios_Feature_46', 'vratios_Feature_29', 'vratios_Feature_83', 'vratios_Feature_96', 'vratios_Feature_95', 'vratios_Feature_84', 'vratios_Feature_94', 'vratios_Feature_4', 'vratios_Feature_45', 'vratios_Feature_3', 'vratios_Feature_2', 'vratios_Feature_42', 'vratios_Feature_5', 'vratios_Feature_39', 'wrythm_Feature_185', 'vratios_Feature_48', 'vratios_Feature_47', 'vratios_Feature_63', 'wrythm_Feature_138', 'vratios_Feature_32', 'wrythm_Feature_162', 'wrythm_Feature_66', 'wrythm_Feature_106', 'wrythm_Feature_130', 'wrythm_Feature_102', 'wrythm_Feature_126', 'wrythm_Feature_54', 'wrythm_Feature_34'],
    'target_b': ['vratios_Feature_46', 'vratios_Feature_29', 'vratios_Feature_96', 'vratios_Feature_45', 'vratios_Feature_83', 'vratios_Feature_4', 'vratios_Feature_95', 'vratios_Feature_84', 'vratios_Feature_3', 'vratios_Feature_94', 'vratios_Feature_2', 'vratios_Feature_5', 'vratios_Feature_42', 'wrythm_Feature_185', 'wrythm_Feature_66', 'wrythm_Feature_162', 'vratios_Feature_48', 'vratios_Feature_47', 'vratios_Feature_63', 'wrythm_Feature_138', 'vratios_Feature_39', 'wrythm_Feature_42', 'wrythm_Feature_106', 'wrythm_Feature_10', 'wrythm_Feature_130', 'wrythm_Feature_114', 'wrythm_Feature_90', 'wrythm_Feature_34', 'vratios_Feature_32', 'wrythm_Feature_186']
}
best_params_dict = {
    'target_w': {
        'colsample_bytree': 0.7,
        'learning_rate': 0.06,
        'max_bin': 332,
        'max_depth': 5, 
        'min_child_samples': 100,  
        'min_child_weight': 5.0,  
        'n_estimators': 500, 
        'num_leaves': 300, 
        'reg_alpha': 10.0,  
        'reg_lambda': 1e-05,
        'subsample': 0.5,  
        'subsample_freq': 3
    }

,
    'target_r': {
        'colsample_bytree': 0.7,
        'learning_rate': 0.06,
        'max_bin': 332,
        'max_depth': 5, 
        'min_child_samples': 100,  
        'min_child_weight': 5.0,  
        'n_estimators': 500, 
        'num_leaves': 300, 
        'reg_alpha': 10.0,  
        'reg_lambda': 1e-05,
        'subsample': 0.5,  
        'subsample_freq': 3
    },
    'target_g': {
        'colsample_bytree': 0.7,
        'learning_rate': 0.06,
        'max_bin': 332,
        'max_depth': 5, 
        'min_child_samples': 100,  
        'min_child_weight': 5.0,  
        'n_estimators': 500, 
        'num_leaves': 300, 
        'reg_alpha': 10.0,  
        'reg_lambda': 1e-05,
        'subsample': 0.5,  
        'subsample_freq': 3
    },
    'target_b': {
        'colsample_bytree': 0.9,  
        'learning_rate': 0.05, 
        'max_bin': 200,  
        'max_depth': 7,  
        'min_child_samples': 10, 
        'min_child_weight': 10.0,  
        'n_estimators': 3000,  
        'num_leaves': 200,  
        'reg_alpha': 1.0,  
        'reg_lambda': 1.0,  
        'subsample': 0.7,  
        'subsample_freq': 1  
    }
}
def preprocess(df, features):
    """
    Preprocess the dataset by handling outliers, transforming data, and normalizing/standardizing.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - features (list): List of features to preprocess.

    Returns:
    - pd.DataFrame: The preprocessed dataframe.
    """
    
    for feature in features:
        
        z_scores = zscore(df[feature])
        threshold = 3
        outliers = abs(z_scores) > threshold
        median_value = df[feature].median()
        df.loc[outliers, feature] = median_value

    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    df[features] = pt.fit_transform(df[features])
    
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    return df

def train(data_path):
    """
    Train LightGBM models on the provided dataset.

    Parameters:
    - data_path (str): Path to the training data file (parquet format).

    Returns:
    - dict: A dictionary containing trained models and their parameters.
    """
    train_data = pd.read_parquet(data_path)
    
    model_info = {}
 
    for target in ['target_w', 'target_r', 'target_g', 'target_b']:
        features = top_features_dict[target]
        params = best_params_dict[target]
        
        # Preprocess the data
        train_data = preprocess(train_data, features)
        
        X = train_data[features]
        y = train_data[target]
        
        lgb_train = lgb.Dataset(X, y)
        
        model = lgb.train(params, lgb_train, num_boost_round=params['n_estimators'])
        
        model_info[target] = model

    return model_info

def inference(data_path, model_info):
    """
    Make predictions using the trained models and round them to the appropriate decimal places.

    Parameters:
    - data_path (str): Path to the test data file (parquet format).
    - model_info (dict): A dictionary containing trained models.

    Returns:
    - pd.DataFrame: A DataFrame with rounded predictions for each target variable.
    """
    test_data = pd.read_parquet(data_path)
    
    preds = pd.DataFrame({'id': test_data['id'], 'Moons': test_data['Moons']})
    

    targets_two_decimal = ['target_b', 'target_w']
    
    for target in ['target_w', 'target_r', 'target_g', 'target_b']:
        features = top_features_dict[target]
        model = model_info[target]
        
        test_X = test_data[features]
        test_pred = model.predict(test_X)
        
        
        if target in targets_two_decimal:
            preds[f'pred_{target}'] = np.round(test_pred, 2)
        else:
            preds[f'pred_{target}'] = np.round(test_pred, 3)
        
    return preds


if __name__ == '__main__':
    train_data_path = '/kaggle/input/crunch/train_data.parquet'
    test_data_path = '/kaggle/input/crunch/X_test.parquet'
    

    def evaluation():

        model_data = train(train_data_path)
        preds = inference(test_data_path, model_data)
        return preds
    preds_wrgb = evaluation()
    
    test_df = pd.read_parquet(test_data_path)
    df_merge = pd.merge(test_df, preds_wrgb, on=['id', 'Moons'])
    scores = []
    for _, gp in df_merge.groupby(['Moons']):
        cur_scores = []
        for target in ['w', 'r', 'g', 'b']:
            cur_i_score = spearmanr(gp[f'target_{target}'].values, gp[f'pred_target_{target}'].values)
            cur_scores.append(cur_i_score.correlation)
        
        scores.append(cur_scores)

    scores = np.array(scores)
    scores_wrgb = np.mean(scores, axis=0)

    print('Spearman correlation scores for each target (w, r, g, b):', scores_wrgb)
    print('Final score:', np.mean(scores_wrgb))