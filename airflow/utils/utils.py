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


def get_monday_of_week(date):
    return date - pd.Timedelta(days=date.weekday())

def preprocessing():
    logging.info("Preprocesamiento Empezó")
    if detect_new_data() == 'branch_a':
      old_data_list = []
      for i in glob.glob('/root/airflow/dags/old_data/*.parquet'):
        dataframe = pd.read_parquet(i)
        old_data_list.append(dataframe)
      df_t = pd.concat(old_data_list)
      df_t.drop_duplicates(inplace=True)
    else:
      new_data_list = []
      for i in glob.glob('/root/airflow/dags/new_data/*.parquet'):
        dataframe = pd.read_parquet(i)
        new_data_list.append(dataframe)
      df_t = pd.concat(new_data_list)
      df_t.drop_duplicates(inplace=True)
    df_c = pd.read_parquet('/root/airflow/dags/other_data/clientes.parquet')
    df_p = pd.read_parquet('/root/airflow/dags/other_data/productos.parquet')

    df = pd.merge(df_t, df_c, on='customer_id', how='left').merge(df_p,on='product_id',how='left')
    df = df.drop_duplicates()

    df = pd.merge(df_t, df_c, on='customer_id', how='left').merge(df_p,on='product_id',how='left')
    df['week'] = pd.to_datetime(df['purchase_date']).apply(get_monday_of_week)


    df_c1 = df.copy()[['customer_id']].value_counts().reset_index()[['customer_id']]
    df_t1 = df.copy()[['week']].value_counts().reset_index()[['week']]
    df_p1 = df.copy()[['product_id']].value_counts().reset_index()[['product_id']]

    df_c1['key'] = '1'
    df_t1['key'] = '1'
    df_p1['key'] = '1'

    df2 = pd.merge(df_c1, df_t1, on='key').merge(df_p1,on='key')
    df2 = df2.drop('key', axis=1)

    df1 = df.groupby(['customer_id','product_id','week'],as_index=False,dropna=False)['items'].sum()

    df3 = pd.merge(df2, df1, on=['customer_id','product_id','week'], how='left')
    df3.fillna(0, inplace=True)
    df3_1 = df3[df3['items'] > 0]
    df3_2 = df3[df3['items'] == 0].sample(len(df3_1))
    df3 = pd.concat([df3_1, df3_2])

    df_t['purchase_date'] = pd.to_datetime(df_t['purchase_date'])
    df_t['week'] = df_t['purchase_date'].apply(get_monday_of_week)

    df_ss = df_t.groupby(['customer_id','week'],as_index=False).size()
    df_ss = df_ss.groupby('customer_id',as_index=False)['size'].mean()
    df_ss.columns = ['customer_id','avg_items']

    df_tc = df_t.groupby('customer_id',as_index=False)['order_id'].size()
    df_tc.columns = ['customer_id','num_orders']

    dfp = df_t.sort_values(['product_id', 'purchase_date']) \
        .groupby(['product_id'])['purchase_date'] \
        .apply(lambda x: pd.Series(sorted(x)).diff().dropna().dt.days.mean() if len(x) > 1 else 0) \
        .reset_index(name='Periodicity')

    df3 = df3.merge(dfp,on='product_id',how='left').merge(df_tc,on='customer_id',how='left').merge(df_ss,on='customer_id',how='left')
    df = pd.merge(df3,df_c,on=['customer_id'],how='left').merge(df_p,on=['product_id'],how='left')
    df.fillna(0, inplace=True)
    df['target'] = np.where(df['items'] > 0,1,0)
    df.drop('items', axis=1, inplace=True)

    df.to_csv('merge_data.csv',index=False)

# def train_model():
#     logging.info("Starting model training")
#     preprocessor = joblib.load('preprocessor.joblib')
#     df = pd.read_csv('merge_data.csv')
#     X_train, X_test, y_train, y_test = train_test_split(df.drop(columns='target'), df['target'], test_size=0.3, random_state=42)
#     X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

#     classifier = XGBClassifier(enable_categorical=True)

#     xgb_pipeline = Pipeline(steps=[
#         ('preprocessor', preprocessor),
#         ('classifier', classifier)
#     ])

#     xgb_pipeline.fit(X_train, y_train)

#     y_pred_xgb = xgb_pipeline.predict(X_val)

#     logging.info("Training completed. Classification report:")
#     logging.info("\n" + classification_report(y_val, y_pred_xgb))



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
    logging.info("Starting model training")
    df = pd.read_csv('merge_data.csv')
    df['week'] = pd.to_datetime(df['week']).dt.isocalendar().week.astype('int')

    X = df.drop(columns='target')
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    cat_cols = ['region_id','zone_id','num_deliver_per_week',
                'num_visit_per_week', 'brand', 'category', 'sub_category', 'segment',
                'package','size','customer_type']
    date_features = ['week']

    for col in cat_cols:
        X_train[col] = X_train[col].astype('category')
        X_val[col] = X_val[col].astype('category')

    best_f1 = -np.inf
    best_pipeline = None
    best_params = None

    def objective(trial):
        nonlocal best_f1, best_pipeline, best_params

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "booster": "gbtree",
            "lambda": trial.suggest_float("lambda", 1e-3, 10.0),
            "alpha": trial.suggest_float("alpha", 1e-3, 10.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "subsample": trial.suggest_float("subsample", 0.3, 1.0),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "n_estimators": 100,
            "use_label_encoder": False,
            "random_state": 42,
            "enable_categorical": True
        }

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        date_pipeline = Pipeline([
            ('date_features', DateFeatureExtractor(date_column='week'))
        ])
        preprocessor = ColumnTransformer([
            ("cat", categorical_pipeline, cat_cols),
            ('date', date_pipeline, date_features)
        ])

        model = xgb.XGBClassifier(**params)
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        full_pipeline.fit(X_train, y_train)
        preds = full_pipeline.predict(X_val)
        f1 = f1_score(y_val, preds)

        # Guardar el mejor pipeline y sus params
        if f1 > best_f1:
            best_f1 = f1
            best_pipeline = pickle.loads(pickle.dumps(full_pipeline))  # Deep copy
            best_params = params.copy()

        # MLflow: solo métricas, params y plot
        with mlflow.start_run(run_name=f"trial_f1_{f1:.4f}"):
            mlflow.log_metric("valid_f1", f1)
            mlflow.log_params(params)
            # Interpretabilidad
            plt.figure(figsize=(8,6))
            xgb.plot_importance(model)
            plt.title("Feature Importance")
            plt.tight_layout()
            plot_path = "feature_importance.png"
            plt.savefig(plot_path)
            plt.close()
            mlflow.log_artifact(plot_path, artifact_path="plots")
            os.remove(plot_path)

        return f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)

    # Optuna plots
    os.makedirs("plots", exist_ok=True)
    fig1 = optuna.visualization.matplotlib.plot_optimization_history(study)
    fig1_path = "plots/optimization_history.png"
    fig1.figure.savefig(fig1_path)
    plt.close(fig1.figure)

    fig2 = optuna.visualization.matplotlib.plot_param_importances(study)
    fig2_path = "plots/param_importances.png"
    fig2.figure.savefig(fig2_path)
    plt.close(fig2.figure)

    with mlflow.start_run(run_name="Optuna_Plots"):
        mlflow.log_artifact(fig1_path, artifact_path="plots")
        mlflow.log_artifact(fig2_path, artifact_path="plots")

    # Guarda el mejor pipeline con pickle
    os.makedirs("models", exist_ok=True)
    with open("models/full_pipeline.pkl", "wb") as f:
        pickle.dump(best_pipeline, f)
    print(f"Mejor F1: {best_f1:.4f} - Pipeline guardado en models/full_pipeline.pkl")
    print("Hiperparámetros del mejor modelo:", best_params)


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

        df = pd.read_csv('/merge_data.csv')
        preprocessor = mlflow.sklearn.load_model(f"runs:/{best_run_id}/preprocessor")
        X_test = df.drop(columns='target')
        y_test = df['target']
        X_test = preprocessor.transform(X_test)

        model_path = '/root/airflow/dags/models/new_best_xgb_model.pkl'
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Use the model for prediction
        predictions = model.predict(X_test)
        current_f1 = f1_score(y_test, predictions)
        if 100 * (best_f1 - current_f1)/current_f1 < 5:
          return 'optimize_model'
        else:
          return 'predictions'
    except:
        return 'optimize_model'


def predictions():
    df_c = pd.read_parquet('/root/airflow/dags/other_data/clientes.parquet')
    df_p = pd.read_parquet('/root/airflow/dags/other_data/productos.parquet')
    df_t = pd.read_parquet('/root/airflow/dags/old_data/transacciones.parquet')

    with open('models/full_pipeline.pkl', 'rb') as f:
        full_pipeline = pickle.load(f)

    df_t['week'] = pd.to_datetime(df_t['purchase_date']).apply(get_monday_of_week)

    df = pd.merge(df_t, df_c, on='customer_id', how='left').merge(df_p, on='product_id', how='left')

    df_c['key'] = '1'
    df_p['key'] = '1'
    df2 = pd.merge(df_p, df_c, on='key').drop('key', axis=1)
    df2['week'] = df['week'].max() + timedelta(days=7)
    df2['week'] = df2['week'].astype(str)
    df['week'] = df['week'].astype(str)

    df1 = df.groupby(['customer_id', 'product_id', 'week'], as_index=False, dropna=False)['items'].sum()
    df3 = pd.merge(df2, df1, on=['customer_id', 'product_id', 'week'], how='left')

    df_ss = df_t.groupby(['customer_id', 'week'], as_index=False).size()
    df_ss = df_ss.groupby('customer_id', as_index=False)['size'].mean()
    df_ss.columns = ['customer_id', 'avg_items']

    df_tc = df_t.groupby('customer_id', as_index=False)['order_id'].size()
    df_tc.columns = ['customer_id', 'num_orders']

    dfp = df_t.sort_values(['product_id', 'purchase_date']) \
        .groupby(['product_id'])['purchase_date'] \
        .apply(lambda x: pd.Series(sorted(x)).diff().dropna().dt.days.mean() if len(x) > 1 else 0) \
        .reset_index(name='Periodicity')

    df3 = df3.merge(dfp, on='product_id', how='left') \
              .merge(df_tc, on='customer_id', how='left') \
              .merge(df_ss, on='customer_id', how='left')

    df3.fillna(0, inplace=True)
    if 'items' in df3.columns:
        df3.drop(columns='items', inplace=True)

    # --- Asegúrate de que las columnas categóricas tengan el tipo correcto ---
    categorical_cols = ['region_id','zone_id','num_deliver_per_week',
                        'num_visit_per_week', 'brand', 'category', 'sub_category', 'segment',
                        'package','size','customer_type']

    for col in categorical_cols:
        df3[col] = df3[col].astype('category')

    # --- Selecciona solo las columnas usadas en entrenamiento ---
    # (Asegúrate de que estas columnas coincidan con las usadas en el pipeline)
    X_test = df3.copy()
    #X_test.to_csv('data_testing.csv', index=False)
    # --- Predice directamente con el pipeline ---
    predictions = full_pipeline.predict(X_test)

    # --- Guarda resultados ---
    output_df = df3.copy()
    output_df['predictions'] = predictions
    output_df = output_df[output_df.predictions==1].drop(columns='predictions')
    output_df.to_csv('models/data_testing.csv', index=False)
    #logging.info("Predictions saved successfully.")
