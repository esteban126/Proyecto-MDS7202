from datetime import timedelta
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss,classification_report,f1_score
from sklearn import set_config
import xgboost as xgb
import numpy as np
import joblib
import logging
import optuna
from optuna.importance import get_param_importances
import mlflow
import os
import pickle
import matplotlib.pyplot as plt
import glob
from xgboost import XGBClassifier

def add_custom_features(df):
    df = df.sort_values(['customer_id', 'product_id', 'week'])
    df['total_weekly_purchase'] = df.groupby(['customer_id', 'week'])['items'].transform('sum')
    df['customer_avg_purchase'] = df.groupby('customer_id')['items'].transform('mean')
    df['recent_purchase_trend'] = df.groupby('customer_id')['target'].transform(lambda x: x.rolling(window=4, min_periods=1).mean())
    df['product_avg_purchase'] = df.groupby('product_id')['items'].transform('mean')
    df['last_purchase_week'] = df.groupby(['customer_id', 'product_id'])['week'].shift(1)
    df['days_since_last_purchase'] = (pd.to_datetime(df['week']) - pd.to_datetime(df['last_purchase_week'])).dt.days
    df['days_since_last_purchase'] = df['days_since_last_purchase'].fillna(-1)
    df['zone_X'] = df['X'].round(1)
    df['zone_Y'] = df['Y'].round(1)

    # Frecuencia promedio de compra por zona
    zone_freq = df.groupby(['zone_X', 'zone_Y'])['target'].mean().reset_index()
    zone_freq = zone_freq.rename(columns={'target': 'zone_avg_purchase_freq'})
    df = df.merge(zone_freq, on=['zone_X', 'zone_Y'], how='left')

    zone_total = df.groupby(['zone_X', 'zone_Y'])['items'].sum().reset_index()
    zone_total = zone_total.rename(columns={'items': 'zone_total_purchases'})
    df = df.merge(zone_total, on=['zone_X', 'zone_Y'], how='left')

    zone_density = df.groupby(['zone_X', 'zone_Y'])['customer_id'].nunique().reset_index()
    zone_density = zone_density.rename(columns={'customer_id': 'zone_customer_density'})
    df = df.merge(zone_density, on=['zone_X', 'zone_Y'], how='left')

    return df

def get_monday_of_week(date):
    return date - pd.Timedelta(days=date.weekday())

def preprocessing():
    logging.info("Preprocesamiento Empezó")
    # Detecta si hay datos nuevos
    if detect_new_data() == 'read_new_data':
        data_list = []
        for i in glob.glob('/root/airflow/dags/new_data/*.parquet'):
            dataframe = pd.read_parquet(i)
            data_list.append(dataframe)
        df_t = pd.concat(data_list)
        df_t.drop_duplicates(inplace=True)
    else:
        data_list = []
        for i in glob.glob('/root/airflow/dags/old_data/*.parquet'):
            dataframe = pd.read_parquet(i)
            data_list.append(dataframe)
        df_t = pd.concat(data_list)
        df_t.drop_duplicates(inplace=True)

    df_c = pd.read_parquet('/root/airflow/dags/other_data/clientes.parquet')
    df_p = pd.read_parquet('/root/airflow/dags/other_data/productos.parquet')

    df = pd.merge(df_t, df_c, on='customer_id', how='left').merge(df_p, on='product_id', how='left')
    df['week'] = pd.to_datetime(df['purchase_date']).apply(get_monday_of_week)

    # Construcción de combinaciones cliente-producto-semana
    df_t1 = df.copy()[['customer_id','week']].drop_duplicates()
    df_p1 = df.copy()[['product_id','customer_id']].drop_duplicates()
    df2 = pd.merge(df_t1, df_p1, on='customer_id').merge(df_c, on='customer_id')

    df1 = df.groupby(['customer_id','product_id','week'], as_index=False, dropna=False)['items'].sum()
    df3 = pd.merge(df2, df1, on=['customer_id','product_id','week'], how='left')
    df3.fillna(0, inplace=True)
    df3_1 = df3[df3['items'] > 0]
    df3_2 = df3[df3['items'] == 0].sample(len(df3_1), random_state=42)
    df3 = pd.concat([df3_1, df3_2])
    df3 = pd.merge(df3, df_p, on='product_id', how='left')
    df3['target'] = np.where(df3['items'] > 0, 1, 0)

    # Feature engineering avanzado
    df3 = add_custom_features(df3)
    df3.fillna(0, inplace=True)
    if 'items' in df3.columns:
        df3.drop(columns='items', inplace=True)

    df3.to_csv('/root/airflow/merge_data.csv', index=False)


def get_best_model(experiment_id):
    runs = mlflow.search_runs(experiment_id)
    best_model_id = runs.sort_values("metrics.valid_f1", ascending=False)["run_id"].iloc[0]
    best_model = mlflow.sklearn.load_model(f"runs:/{best_model_id}/model")
    return best_model

# ---------- Clase para transformar fechas ----------
class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_column='week'):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['month'] = pd.to_datetime(X_copy[self.date_column]).dt.month
        X_copy['week1'] = pd.to_datetime(X_copy[self.date_column]).dt.isocalendar().week
        return X_copy[['month', 'week1']]

def optimize_model():

    df = pd.read_csv('/root/airflow/merge_data.csv')

    # Split temporal
    df = df.sort_values('week')
    weeks = df['week'].sort_values().unique()
    n = len(weeks)
    train_weeks = weeks[:int(0.7*n)]
    val_weeks = weeks[int(0.7*n):int(0.85*n)]
    test_weeks = weeks[int(0.85*n):]

    train = df[df['week'].isin(train_weeks)]
    val = df[df['week'].isin(val_weeks)]
    test = df[df['week'].isin(test_weeks)]

    feature_cols = [col for col in df.columns if col not in ['target']]
    target_col = 'target'

    X_train = train[feature_cols]
    y_train = train[target_col]
    X_val = val[feature_cols]
    y_val = val[target_col]
    X_test = test[feature_cols]
    y_test = test[target_col]

    cat_cols = [
        'brand', 'category', 'sub_category', 'segment', 'package', 'customer_type',
        'zone_X', 'zone_Y'
    ]
    num_cols = [
        'size', 'num_deliver_per_week',
        'total_weekly_purchase', 'customer_avg_purchase', 'recent_purchase_trend',
        'product_avg_purchase', 'days_since_last_purchase',
        'zone_avg_purchase_freq', 'zone_total_purchases', 'zone_customer_density'
    ]
    date_features = ['week']

    def objective(trial):
        imputer_strategy = trial.suggest_categorical("imputer_strategy", ["mean", "median"])
        scaler_type = trial.suggest_categorical("scaler", ["standard", "minmax"])
        encoder_type = trial.suggest_categorical("encoder", ["onehot", "ordinal"])

        scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy=imputer_strategy)),
            ("scaler", scaler)
        ])

        if encoder_type == "onehot":
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        else:
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder)
        ])

        date_pipeline = Pipeline([
            ('date_features', DateFeatureExtractor(date_column='week'))
        ])

        preprocessor = ColumnTransformer([
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
            ("date", date_pipeline, date_features)
        ])

        clf = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            gamma=trial.suggest_float("gamma", 0, 5),
            reg_alpha=trial.suggest_float("reg_alpha", 0.0, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.0, 1.0),
            use_label_encoder=False,
            eval_metric="aucpr",
            scale_pos_weight=(len(y_train) / y_train.sum()),
            random_state=42,
            n_jobs=-1
        )

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        f1 = f1_score(y_val, preds)
        return f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)

    # Entrenamiento final con mejores hiperparámetros
    best_params = study.best_params
    scaler = StandardScaler() if best_params["scaler"] == "standard" else MinMaxScaler()
    if best_params["encoder"] == "onehot":
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=best_params["imputer_strategy"])),
        ("scaler", scaler)
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", encoder)
    ])
    date_pipeline = Pipeline([
        ('date_features', DateFeatureExtractor(date_column='week'))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols),
        ("date", date_pipeline, date_features)
    ])

    clf = XGBClassifier(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        subsample=best_params["subsample"],
        colsample_bytree=best_params["colsample_bytree"],
        gamma=best_params["gamma"],
        reg_alpha=best_params["reg_alpha"],
        reg_lambda=best_params["reg_lambda"],
        use_label_encoder=False,
        eval_metric="aucpr",
        scale_pos_weight=(len(y_train) / y_train.sum()),
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    print("Reporte de Clasificación del Modelo Optimizado:")
    print(classification_report(y_test, y_pred))

    # Guarda el pipeline
    import os
    os.makedirs("/root/airflow/models", exist_ok=True)
    with open("/root/airflow/models/full_pipeline.pkl", "wb") as f:
        pickle.dump(pipe, f)

def detect_new_data():
    old_data_dir = '/root/airflow/dags/old_data/'
    new_data_dir = '/root/airflow/dags/new_data/'
    current_files = len(set(os.listdir(old_data_dir)))
    new_files = len(set(os.listdir(new_data_dir)))
    new_data_files = new_files - current_files
    print(new_data_files)
    if new_data_files >= 0:
      return "read_new_data"
    else:
      return "dont_predict"

def read_new_data():
    old_data_list = []
    new_data_list = []
    for i in glob.glob('/root/airflow/dags/new_data/*.parquet'):
      dataframe = pd.read_parquet(i)
      old_data_list.append(dataframe)
      df_nt = pd.concat(old_data_list)
    for i in glob.glob('/root/airflow/dags/old_data/*.parquet'):
      dataframe = pd.read_parquet(i)
      new_data_list.append(dataframe)
      df_ct = pd.concat(new_data_list)
    df_t = pd.concat([df_nt,df_ct])
    df_t.drop_duplicates(inplace=True)
    start_date = pd.to_datetime(df_nt.purchase_date.min()).strftime('%Y%m%d')
    end_date   = pd.to_datetime(df_nt.purchase_date.max()).strftime('%Y%m%d')
    df_nt.to_parquet(f'/root/airflow/dags/old_data/transacciones_{start_date}_{end_date}.parquet')

def branch_use_current_params():
    try:
        experiment_name = "XGBoost_Optuna_Optimization"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        runs = mlflow.search_runs(experiment.experiment_id)
        best_f1 = runs['metrics.valid_f1'].max()
        best_run_id = runs[runs['metrics.valid_f1'] == best_f1].run_id.iloc[0]

        # Carga datos completos
        df = pd.read_csv('/root/airflow/merge_data.csv')
        preprocessor = mlflow.sklearn.load_model(f"runs:/{best_run_id}/preprocessor")

        # Identifica la última semana
        last_week = df['week'].max()
        mask_new = df['week'] == last_week
        mask_ref = df['week'] != last_week

        # Separa referencia y nueva data
        X_ref = df[mask_ref].drop(columns='target')
        X_test_new = df[mask_new].drop(columns='target')

        # Si no hay suficiente data de referencia o nueva, no hacer drift
        if X_ref.shape[0] < 10 or X_test_new.shape[0] < 10:
            print("No hay suficiente data para comparar drift.")
            return 'predictions'

        # Preprocesa ambas
        X_ref_proc = preprocessor.transform(X_ref)
        X_test_proc = preprocessor.transform(X_test_new)

        # Drift detection
        cd = MMDDrift(X_ref_proc, p_val=0.05)
        preds = cd.predict(X_test_proc)
        is_drift = preds['data']['is_drift']
        p_value = preds['data']['p_val']
        print(f"Drift detected: {is_drift}, p-value: {p_value}")

        if is_drift:
            return 'optimize_model'
        else:
            return 'predictions'
    except Exception as e:
        print("Error:", e)
        return 'optimize_model'

def predictions():

    df_c = pd.read_parquet('/root/airflow/dags/other_data/clientes.parquet')
    df_p = pd.read_parquet('/root/airflow/dags/other_data/productos.parquet')
    df_t = pd.read_parquet('/root/airflow/dags/old_data/transacciones.parquet')

    with open('/root/airflow/models/full_pipeline.pkl', 'rb') as f:
        full_pipeline = pickle.load(f)

    df_t['week'] = pd.to_datetime(df_t['purchase_date']).apply(get_monday_of_week)
    df = pd.merge(df_t, df_c, on='customer_id', how='left').merge(df_p, on='product_id', how='left')

    # Construcción de combinaciones cliente-producto-semana
    df_t1 = df.copy()[['customer_id','week']].drop_duplicates()
    df_p1 = df.copy()[['product_id','customer_id']].drop_duplicates()
    df2 = pd.merge(df_t1, df_p1, on='customer_id').merge(df_c, on='customer_id')

    df1 = df.groupby(['customer_id','product_id','week'], as_index=False, dropna=False)['items'].sum()
    df3 = pd.merge(df2, df1, on=['customer_id','product_id','week'], how='left')
    df3.fillna(0, inplace=True)
    df3_1 = df3[df3['items'] > 0]
    df3_2 = df3[df3['items'] == 0]
    df3 = pd.concat([df3_1, df3_2])
    df3 = pd.merge(df3, df_p, on='product_id', how='left')
    df3['target'] = np.where(df3['items'] > 0, 1, 0)

    # Feature engineering avanzado
    df3 = add_custom_features(df3)
    df3.fillna(0, inplace=True)
    if 'items' in df3.columns:
        df3.drop(columns='items', inplace=True)

    # Predicción
    X_pred = df3.copy()
    predictions = full_pipeline.predict(X_pred)

    output_df = df3.copy()
    output_df['predictions'] = predictions
    output_df = output_df[output_df.predictions == 1].drop(columns='predictions')
    output_df.to_csv('/root/airflow/models/data_testing.csv', index=False)

