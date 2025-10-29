# src/optimization.py (actualizar)
import optuna
import lightgbm as lgb
import polars as pl
import numpy as np
import logging
import yaml
import os
import datetime
from lightgbm import LGBMClassifier
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit


logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            logger.info(f"✅ Loaded configuration from {config_path}")
            return config
    except Exception as e:
        logger.error(f"❌ Error loading config from {config_path}: {e}")
        return {}

config_path = "config-dev.yml"
config = load_config(config_path)
MES_TRAIN = config["MES_TRAIN"]
MES_VALIDACION = config["MES_VALIDACION"]
BUCKET = config["BUCKET"]
STUDY_NAME = config["STUDY_NAME"]
DATASET_FE_PATH = config["DATASET_FE_PATH"]
GANANCIA_ACIERTO = config["GANANCIA_ACIERTO"]
COSTO_ESTIMULO = config["COSTO_ESTIMULO"]
FINAL_PREDICT = config["FINAL_PREDICT"]
MES_TEST = config["MES_TEST"]
FINAL_TRAIN = config["FINAL_TRAIN"]
SEMILLA = config["SEMILLA"]
MODELOS_PATH = config["MODELOS_PATH"]

debug = False
run_bayesian_optimization = False
submit = False
submission_number = 1

estudio = 1

threshold = 0.15





df = pl.read_csv(os.path.join(BUCKET,DATASET_FE_PATH))

df = df.with_columns(
    pl.lit(1.0).alias('clase_peso')
).with_columns(
    pl.when(pl.col('clase_ternaria') == 'BAJA+2')
    .then(pl.lit(1.00002))
    .otherwise(pl.col('clase_peso'))
    .alias('clase_peso')
).with_columns(
    pl.when(pl.col('clase_ternaria') == 'BAJA+1')
    .then(pl.lit(1.00001))
    .otherwise(pl.col('clase_peso'))
    .alias('clase_peso')
)

clientes_test = df.filter(pl.col('foto_mes') == MES_VALIDACION)["numero_de_cliente"]

clientes_predict = df.filter(pl.col('foto_mes') == FINAL_PREDICT)["numero_de_cliente"]


df = df.drop(['numero_de_cliente','tmobile_app','mplazo_fijo_dolares'])

df_train = df.filter(pl.col('foto_mes').is_in(MES_TRAIN))
df_test = df.filter(pl.col('foto_mes') == MES_VALIDACION)
df_predict = df.filter(pl.col('foto_mes') == FINAL_PREDICT)

df_train_predict = df.filter(pl.col('foto_mes').is_in(FINAL_TRAIN))

df_train = df_train.drop(['foto_mes'])
df_test = df_test.drop(['foto_mes'])
df_predict = df_predict.drop(['foto_mes'])
df_train_predict = df_train_predict.drop(['foto_mes'])

df_train_weight = df_train['clase_peso']
df_train_clase_binaria_baja = df_train['clase_binaria']
df_test_clase_binaria_baja = df_test['clase_binaria']

df_predict_clase_binaria_baja = df_train_predict['clase_binaria']
df_train_predict_weight = df_train_predict['clase_peso']

df_train = df_train.drop(['clase_binaria','clase_peso'])
df_train_predict = df_train_predict.drop(['clase_binaria','clase_peso'])
df_test = df_test.drop(['clase_binaria','clase_peso'])
df_predict = df_predict.drop(['clase_binaria','clase_peso'])

df_val = df_train(pl.col('foto_mes').is_in(MES_VALIDACION))

def lgb_gan_eval(y_pred, data):
    weight = data.get_weight()
    ganancia = np.where(weight == 1.00002, GANANCIA_ACIERTO, 0) - np.where(weight < 1.00002, COSTO_ESTIMULO, 0)
    ganancia = ganancia[np.argsort(y_pred)[::-1]]
    ganancia = np.cumsum(ganancia)

    return 'gan_eval', np.max(ganancia) , True

sss_opt = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=SEMILLA[1])
resultados_medias = []

def objective(trial, X, y, weight, sss) -> float:
    """
    Parameters:
    trial: trial de optuna
    df: dataframe con datos
  
    Description:
    Función objetivo que maximiza ganancia en mes de validación.
    Utiliza configuración YAML para períodos y semilla.
    Define parametros para el modelo LightGBM
    Preparar dataset para entrenamiento y validación
    Entrena modelo con función de ganancia personalizada
    Predecir y calcular ganancia
    Guardar cada iteración en JSON
  
    Returns:
    float: ganancia total
    """
    # Hiperparámetros a optimizar
    num_leaves = trial.suggest_int('num_leaves', 8, 80)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.4)
    max_depth = trial.suggest_int("max_depth", -1, 50)
    min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 1, 1000)
    feature_fraction = trial.suggest_float('feature_fraction', 0.1, 1.0)
    max_bin = trial.suggest_int('max_bin', 255, 500)
    num_iterations = trial.suggest_int('num_iterations', 100, 500)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'rf',
        'first_metric_only': True,
        'boost_from_average': True,
        'feature_pre_filter': False,
        'max_bin': max_bin,
        'max_depth': max_depth,
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'min_data_in_leaf': min_data_in_leaf,
        'feature_fraction': feature_fraction,
        'seed': SEMILLA[1],
        'verbose': -1,
        'num_iterations': num_iterations
        }

    train_data = lgb.Dataset(X,
                                label=y,
                                weight=weight)
  

    if isinstance(MES_TRAIN, list):
        df_train = df[df['foto_mes'].isin(MES_TRAIN)]
    else:
        df_train = df[df['foto_mes'] == MES_TRAIN]
  
    df_val = df[df['foto_mes'] == MES_VALIDACION]
  
 

    model = lgb.train(
        params,
        df_train,
        #valid_sets=[val_data],
        feval=lgb_gan_eval,  # Función de ganancia personalizada
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
  
    # Predecir y calcular ganancia
    y_pred_proba = model.predict(df_val)
  
    #COMO OBSERVARON USO LA MISMA FUNCION DE GANANCIA PARA ENTRENAR Y VALIDAR, CAMBIEN A GUSTO!!!!!
    _, ganancia_total, _ = ganancia_evaluator(y_pred_proba, df_val)

    # Guardar cada iteración en JSON
    guardar_iteracion(trial, ganancia_total)
  
    logger.info(f"Trial {trial.number}: Ganancia = {ganancia_total:,.0f}")
  
    return ganancia_total





storage_name = f"sqlite:////{os.path.join(BUCKET,STUDY_NAME)}.db"
study_name = f"exp_comp_2_{estudio}_lgbm-opt"

study = optuna.create_study(
    direction="maximize",
    study_name=study_name,
    storage=storage_name,
    load_if_exists=True,
)

if run_bayesian_optimization:
  study.optimize(lambda trial: objective(trial, df_train, df_train_clase_binaria_baja, df_train_weight, sss_opt), n_trials=50)

def aplicar_undersampling(df: pd.DataFrame, ratio: float, random_state: int = None) -> pd.DataFrame:
    pass



## config basico logging
os.makedirs(f"{BUCKET}/logs", exist_ok=True)

fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
nombre_log = f"log_{STUDY_NAME}_{fecha}.log"

log_path =os.path.join(f"{BUCKET}/logs/", nombre_log)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8-sig"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)




def build_and_save_models(semillas, train_dataset, y_target, weight, is_test):
  # Convert Polars DataFrames to Pandas DataFrames
  train_dataset_pd = train_dataset.to_pandas()
  y_target_pd = y_target.to_pandas()
  weight_pd = weight.to_pandas()

  train_data = lgb.Dataset(train_dataset_pd,
                              label=y_target_pd,
                              weight=weight_pd)

  modelos = {}
  print(f"Construimos los models para las semillas : {semillas}")

  for seed in semillas:
    print(f"Semilla: {seed}")

    params = {
            'objective': 'binary',
              'metric': 'auc',
              'boosting_type': 'rf',
              'first_metric_only': True,
              'boost_from_average': True,
              'feature_pre_filter': False,
              'max_bin': 31,
              'seed': seed,
              'verbose': -1
        }


    if run_bayesian_optimization:
      best_iter = study.best_trial.user_attrs["best_iter"]
      new_params = study.best_trial.params
      new_params['n_estimators'] = best_iter
    else:
      # Corrida 1 y 3 son pésimas..
      # Corrida 2 : new_params = {'num_leaves': 65, 'learning_rate': 0.22174535843285384, 'max_depth': 40, 'min_data_in_leaf': 161, 'feature_fraction': 0.5328801616449493, 'bagging_fraction': 0.1641705775874533, 'min_child_samples': 41, 'n_estimators': 403}
      # Corrida 1 : new_params = {'num_leaves': 93, 'learning_rate': 0.15548003319593617, 'max_depth': 39, 'min_data_in_leaf': 759, 'feature_fraction': 0.119199901114561, 'bagging_fraction': 0.12291684731958094, 'min_child_samples': 30, 'n_estimators': 110}

      # Corrida 4 : new_params = {'num_leaves': 54, 'learning_rate': 0.166278661717272, 'max_depth': 42, 'min_data_in_leaf': 310, 'feature_fraction': 0.45780488981801093, 'bagging_fraction': 0.2029691560601475, 'min_child_samples': 62, 'max_bin': 416, 'num_iterations': 343}
      # Corrida 5 : new_params = {'num_leaves': 70, 'learning_rate': 0.16924391708150185, 'max_depth': 9, 'min_data_in_leaf': 510, 'feature_fraction': 0.5552248387271188, 'max_bin': 363, 'num_iterations': 372}

      # Corrida 3 : new_params = {'num_leaves': 73, 'learning_rate': 0.2497842951354348, 'max_depth': 12, 'min_data_in_leaf': 633, 'feature_fraction': 0.17951553564916345, 'bagging_fraction': 0.7869774609783072, 'min_child_samples': 58, 'n_estimators': 582}
      new_params = {'num_leaves': 54, 'learning_rate': 0.166278661717272, 'max_depth': 42, 'min_data_in_leaf': 310, 'feature_fraction': 0.45780488981801093, 'bagging_fraction': 0.2029691560601475, 'min_child_samples': 62, 'max_bin': 416, 'num_iterations': 343}

    params.update(new_params)
    model = lgb.train(params,
                  train_data)

    modelos[seed] = model
    if is_test:
      model.save_model(MODELOS_PATH + f'lgb_test_{seed}_{submission_number}.txt')
    else:
      model.save_model(MODELOS_PATH + f'lgb_predict_{seed}_{submission_number}.txt')
  return modelos

if __name__ == "__main__":
    main()
