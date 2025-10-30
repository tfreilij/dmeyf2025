# src/optimization.py (actualizar)
from token import SEMI
import optuna
import lightgbm as lgb
import polars as pl
import numpy as np
import logging
import yaml
import os
import datetime
from lightgbm import LGBMClassifier
from load_config import Config


logger = logging.getLogger(__name__)

def drop_columns(df : pl.DataFrame):


    col_drops = [
          "numero_de_cliente", "active_quarter", "clase_ternaria",
          "cliente_edad", "cliente_antiguedad",
          "Visa_fultimo_cierre", "Master_fultimo_cierre",
          "Visa_Fvencimiento", "Master_Fvencimiento"
    ]

    df = df.drop(['numero_de_cliente','tmobile_app','mplazo_fijo_dolares'])
    return df


def ganancia_evaluator(y_pred, y_true) -> float:

    y_true = y_true.get_label()
  
    # Convertir a DataFrame de Polars para procesamiento eficiente
    df_eval = pl.DataFrame({'y_true': y_true,'y_pred_proba': y_pred})
  
    # Ordenar por probabilidad descendente
    df_ordenado = df_eval.sort('y_pred_proba', descending=True)
  
    # Calcular ganancia individual para cada cliente
    df_ordenado = df_ordenado.with_columns([pl.when(pl.col('y_true') == 1).then(GANANCIA_ACIERTO).otherwise(-COSTO_ESTIMULO).alias('ganancia_individual')])
  
    # Calcular ganancia acumulada
    df_ordenado = df_ordenado.with_columns([pl.col('ganancia_individual').cum_sum().alias('ganancia_acumulada')])
  
    # Encontrar la ganancia máxima
    ganancia_maxima = df_ordenado.select(pl.col('ganancia_acumulada').max()).item()
  
    return 'ganancia', ganancia_maxima, True


def generate_clase_peso(df : pl.DataFrame):

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

    return df

def generate_clase_binaria(df : pl.DataFrame):

    df = df.with_columns(pl.lit(0).alias('clase_binaria'))

    df = df.with_columns(
        pl.when(pl.col('clase_ternaria').is_in(['BAJA+2','BAJA+1'])).then(pl.lit(1)).otherwise(pl.lit(0)).alias('clase_binaria')
    )

    df = df.drop(['clase_ternaria'])

    return df

def lgb_gan_eval(y_pred, data):
    weight = data.get_weight()
    ganancia = np.where(weight == 1.00002, GANANCIA_ACIERTO, 0) - np.where(weight < 1.00002, COSTO_ESTIMULO, 0)
    ganancia = ganancia[np.argsort(y_pred)[::-1]]
    ganancia = np.cumsum(ganancia)

    return 'gan_eval', np.max(ganancia) , True


config = Config()
MES_TRAIN = config["MES_TRAIN"]
MES_VALIDACION = config["MES_VALIDACION"]
BUCKET = config["BUCKET"]
STUDY_NAME = config["STUDY_NAME"]
DATASET_FE_FILE = config["DATASET_FE_FILE"]
GANANCIA_ACIERTO = config["GANANCIA_ACIERTO"]
COSTO_ESTIMULO = config["COSTO_ESTIMULO"]
FINAL_PREDICT = config["FINAL_PREDICT"]
MES_TEST = config["MES_TEST"]
FINAL_TRAIN = config["FINAL_TRAIN"]
SEMILLA = config["SEMILLA"]
MODELOS_PATH = config["MODELOS_PATH"]
THRESHOLD = config["THRESHOLD"]
SUBMISSION_NUMBER = config["SUBMISSION_NUMBER"]

debug = False
run_bayesian_optimization = False
submit = False
train_test_models = True

df = pl.read_csv(os.path.join(BUCKET,DATASET_FE_FILE))

df = generate_clase_peso(df)

df = generate_clase_binaria(df)

clientes_test = df.filter(pl.col('foto_mes') == MES_VALIDACION)["numero_de_cliente"]
clientes_predict = df.filter(pl.col('foto_mes') == FINAL_PREDICT)["numero_de_cliente"]

df = drop_columns(df)

df_train = df.filter(pl.col('foto_mes').is_in(MES_TRAIN))
df_test = df.filter(pl.col('foto_mes') == MES_VALIDACION)
df_predict = df.filter(pl.col('foto_mes') == FINAL_PREDICT)

df_train_predict = df.filter(pl.col('foto_mes').is_in(FINAL_TRAIN))

df_val = df_train.filter(pl.col('foto_mes') == MES_VALIDACION)

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


resultados_medias = []

def objective(trial, X : pl.DataFrame, y : pl.DataFrame , weight : pl.DataFrame) -> float:

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

    X_pd = X.to_pandas()
    y_pd = y.to_pandas()
    weight_pd = weight.to_pandas()

    train_data = lgb.Dataset(X_pd,
                                label=y_pd,
                                weight=weight_pd)

    model = lgb.train(
        params,
        train_data,
        #valid_sets=[val_data],
        feval=lgb_gan_eval,  # Función de ganancia personalizada
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
  
    y_pred_proba = model.predict(df_val)
  
    _, ganancia_total, _ = ganancia_evaluator(y_pred_proba, df_val)

    logger.info(f"Trial {trial.number}: Ganancia = {ganancia_total:,.0f}")
  
    return ganancia_total



storage_name = f"sqlite:////{os.path.join(BUCKET,STUDY_NAME)}.db"
study_name = STUDY_NAME

study = optuna.create_study(
    direction="maximize",
    study_name=study_name,
    storage=storage_name,
    load_if_exists=True,
)

if run_bayesian_optimization:
  study.optimize(lambda trial: objective(trial, df_train, df_train_clase_binaria_baja, df_train_weight, sss_opt), n_trials=50)

def aplicar_undersampling(df: pl.DataFrame, ratio: float, random_state: int = None) -> pl.DataFrame:
    pass

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

def build_and_save_models(semillas : list, train_dataset : pl.DataFrame, y_target : pl.DataFrame , weight : pl.DataFrame, is_test) -> list:
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
      new_params = {'num_leaves': 54, 'learning_rate': 0.166278661717272, 'max_depth': 42, 'min_data_in_leaf': 310, 'feature_fraction': 0.45780488981801093, 'bagging_fraction': 0.2029691560601475, 'min_child_samples': 62, 'max_bin': 416, 'num_iterations': 343}

    params.update(new_params)
    model = lgb.train(params,
                  train_data)

    modelos[seed] = model
    if is_test:
      model.save_model(MODELOS_PATH + f'lgb_test_{seed}_{SUBMISSION_NUMBER}.txt')
    else:
      model.save_model(MODELOS_PATH + f'lgb_predict_{seed}_{SUBMISSION_NUMBER}.txt')
  return modelos

def ganancia_prob(y_pred, y_true, threshold,prop = 1):
  ganancia = np.where(y_true == 1, GANANCIA_ACIERTO, 0) - np.where(y_true == 0, COSTO_ESTIMULO, 0)
  return ganancia[y_pred >= threshold].sum() / prop


def binarize_predictions(y_pred):
    return np.where(y_pred >= THRESHOLD, 1, 0)


def build_predictions(clientes, modelos, dataset, threshold,y_true=None):
  predicciones = {}
  for seed,model in modelos.items():
    if seed in SEMILLA:
      print(f"Semilla: {seed}")
      predictions = model.predict(dataset)
      predicciones[seed] = predictions
      if y_true is not None:
        print(f"Ganancias de Modelo con semilla {seed}:", ganancia_prob(predictions, y_true,threshold))

  mean_predictions = np.mean(list(predicciones.values()), axis=0)
  return pl.DataFrame({'numero_de_cliente': clientes, 'Predicted': binarize_predictions(mean_predictions,threshold)})


print(SEMILLA)
test_models = {}
for seed in SEMILLA:
  model_file_path = MODELOS_PATH + f'lgb_test_{seed}_{SUBMISSION_NUMBER}.txt'
  if os.path.exists(model_file_path):
    print(f"Cargamos el modelo de Test de la submission {SUBMISSION_NUMBER} para la semilla {seed}")
    booster = lgb.Booster(model_file=model_file_path)
    test_models[seed] = booster
    train_test_models = False
  else:
    print(f"El modelo de Test para la semilla {seed} no existe en {model_file_path}. Se entrenará.")

if train_test_models:
  test_models = build_and_save_models(SEMILLA, df_train, df_train_clase_binaria_baja, df_train_weight,is_test=True)


train_predict_models = True

predict_models = {}
for seed in SEMILLA:
  model_file_path = MODELOS_PATH + f'lgb_predict_{seed}_{SUBMISSION_NUMBER}.txt'
  if os.path.exists(model_file_path):
    print(f"Cargamos el modelo de Predicción de la submission {SUBMISSION_NUMBER} para la semilla {seed}")
    modelo_anterior = lgb.Booster(model_file=model_file_path)
    predict_models[seed] = modelo_anterior
    train_predict_models = False
  else:
    print(f"El modelo de Predicción para la semilla {seed} no existe en {model_file_path}. Se entrenará.")

if train_predict_models:
  predict_models = build_and_save_models(SEMILLA,df_train_predict,df_predict_clase_binaria_baja, df_train_predict_weight, is_test=False)


test_predictions = build_predictions(clientes_test, test_models, df_test, threshold=THRESHOLD, y_true=df_test_clase_binaria_baja)

kaggle_predictions = build_predictions(clientes_predict, predict_models, df_predict, threshold=THRESHOLD, y_true=None)

if submit:
  kaggle_predictions.write_csv(BUCKET + f"predictions_{SUBMISSION_NUMBER}.csv")