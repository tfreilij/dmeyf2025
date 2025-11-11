# src/optimization.py (actualizar)
from token import SEMI
import optuna
import lightgbm as lgb
import polars as pl
import numpy as np
import logging
import os
import datetime
from load_config import Config

logger = logging.getLogger(__name__)

## VARIABLES DE ENTORNO Y CONFIGURACION
config = Config()
MES_TRAIN = config["MES_TRAIN"]
BUCKETS = config["BUCKETS"]
BUCKET_ORIGIN = config["BUCKET_ORIGIN"]
BUCKET_TARGET = config["BUCKET_TARGET"]
SUBMIT = config["SUBMIT"]
MES_VALIDACION = config["MES_VALIDACION"]
STUDY_NAME = config["STUDY_NAME"]
GANANCIA_ACIERTO = config["GANANCIA_ACIERTO"]
COSTO_ESTIMULO = config["COSTO_ESTIMULO"]
FINAL_PREDICT = config["FINAL_PREDICT"]
MES_TEST = config["MES_TEST"]
FINAL_TRAIN = config["FINAL_TRAIN"]
SEMILLA = config["SEMILLA"]
SUBMISSION_NUMBER = config["SUBMISSION_NUMBER"]
UNDERSAMPLE_FRACTION = config["UNDERSAMPLING_FRACTION"]
RUN_BAYESIAN_OPTIMIZATION = config["RUN_BAYESIAN_OPTIMIZATION"]

submission_number = 1


## DIRECTORIOS PARA LOGGING Y MODELOS
bucket_target = os.path.join(BUCKETS,BUCKET_TARGET)
modelos_directory = os.path.join(bucket_target,"modelos")

nombre_log = f"log_{STUDY_NAME}_{datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")}.log"
log_directory = os.path.join(bucket_target,"log")

os.makedirs(log_directory, exist_ok=True)
os.makedirs(modelos_directory, exist_ok=True)

log_path = os.path.join(log_directory, nombre_log)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8-sig"),
        logging.StreamHandler()
    ]
)

## DROPEAR COLUMNAS QUE NO VAN PARA ENTRENAMIENTO
def drop_columns(df : pl.DataFrame):
    logger.info("Drop columns")

    col_drops = ["Visa_Finiciomora","Visa_Finiciomora",
          "Visa_fultimo_cierre", "Master_fultimo_cierre",
          "Visa_Fvencimiento", "Master_Fvencimiento"
      ]

    df = df.drop(['numero_de_cliente','tmobile_app','mprestamos_personales'])
    df = df.drop(col_drops)
    return df

def ganancia_prob(y_pred, y_true, threshold,prop = 1):
  ganancia = np.where(y_true == 1, GANANCIA_ACIERTO, 0) - np.where(y_true == 0, COSTO_ESTIMULO, 0)
  return ganancia[y_pred >= threshold].sum() / prop

## SE ARMAN LAS PREDICCIONES PROMEDIADAS
def build_predictions(clientes, modelos, dataset):
  predicciones = {}
  logger.info(f"Build predictions")
  for seed,model in modelos.items():
    if seed in SEMILLA:
      logger.info(f"Semilla: {seed}")
      predictions = model.predict(dataset)
      predicciones[seed] = predictions

  mean_predictions = np.mean(list(predicciones.values()), axis=0)
  return pl.DataFrame({'numero_de_cliente': clientes, 'Predicted': mean_predictions})

## SE ARMAN LAS PREDICCIONES PARA EL TARGET
def build_final_predictions(clientes_predict, predict_models, df_predict, n_envios):
  mean_predictions = build_predictions(clientes_predict, predict_models, df_predict)
  sorted_mean_predictions = mean_predictions.sort('Predicted', descending=True)
  final_predictions = sorted_mean_predictions.with_columns(
        (pl.arange(0, sorted_mean_predictions.height) < n_envios)
        .cast(pl.Int8)
        .alias("Prediction")
    )
    
  return final_predictions.select(["numero_de_cliente", "Prediction"])


## SE APLICAR UNDERSAMPLING. SÓLO DEBERÍA USARSE EN EL DF_TRAIN
def undersample_df(df: pl.DataFrame, fraction) -> pl.DataFrame:
  logger.info(f"Undersampling Continuas with fraction : {fraction} , DF shape : {df.shape}")
  
  clientes_solo_continuas = df.group_by("numero_de_cliente").agg(n_bajas=pl.col("clase_binaria").sum()).filter(pl.col("n_bajas") == 0)
  clientes_solo_continuas_undersampled = clientes_solo_continuas.sample(fraction=1-fraction, seed=1000)
  df = df.filter(~pl.col('numero_de_cliente').is_in(clientes_solo_continuas_undersampled["numero_de_cliente"]))
  logger.info(f"DF shape after undersampling: {df.shape}")
  return df

## SE ORDENAN DE MAYOR A MENOR LAS PROBABILIDADES Y SE BUSCA LA MAXIMA GANANCIA JUNTO A LA CANTIDAD DE ENVIOS CORRESPONDIENTES
def ganancia_evaluator(y_pred, y_true) -> float:

  logger.info("Ganancia evaluator")
  logger.info(f"Y_true : {y_true}")
  logger.info(f"Y_pred : {y_pred}")
  y_true = y_true

  df_eval = pl.DataFrame({'y_true': y_true,'y_pred_proba': y_pred["Predicted"]})
  df_ordenado = df_eval.sort('y_pred_proba', descending=True)
  df_ordenado = df_ordenado.with_columns([
      pl.when(pl.col('y_true') == 1)
        .then(GANANCIA_ACIERTO)
        .otherwise(-COSTO_ESTIMULO)
        .alias('ganancia_individual'), pl.lit(1).alias('indice')
  ])
  
  df_ordenado = df_ordenado.with_columns([
      pl.col('ganancia_individual').cum_sum().alias('ganancia_acumulada'),
      pl.col('indice').cum_sum().alias('indice_acumulado')
  ])
  
  ganancia_maxima_valor = df_ordenado.select(pl.col('ganancia_acumulada').max()).item()
  envios_ganancia_maxima = df_ordenado.filter(pl.col('ganancia_acumulada') == ganancia_maxima_valor).head(1)

  ganancia_maxima = envios_ganancia_maxima.select('ganancia_acumulada')
  cantidad_envios = envios_ganancia_maxima.select('indice_acumulado')
  logger.info(f"GANANCIA MAXIMA : {ganancia_maxima} , {cantidad_envios}")
  return ganancia_maxima, cantidad_envios

def generate_clase_peso(df : pl.DataFrame):

    logger.info("Generate Clase Peso")
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

## SE BINARIZA LA CLASE OBJETIVO.
def generate_clase_binaria(df : pl.DataFrame):

    df = df.with_columns(pl.lit(0).alias('clase_binaria'))

    df = df.with_columns(
        pl.when(pl.col('clase_ternaria').is_in(['BAJA+2','BAJA+1'])).then(pl.lit(1)).otherwise(pl.lit(0)).alias('clase_binaria')
    )

    return df

## SE ARMA EL MODELO Y DE SER POSIBLE SE PERSISTE PARA PODER USARLO PARA OTRA PREDICCIÓN.
def build_and_save_models(study, semillas : list, train_dataset : pl.DataFrame, y_target : pl.DataFrame , weight : pl.DataFrame, is_test, run_bayesian_optimization) -> list:

  train_dataset_pd = train_dataset.to_pandas()
  y_target_pd = y_target.to_pandas()
  weight_pd = weight.to_pandas()

  train_data = lgb.Dataset(train_dataset_pd,
                              label=y_target_pd,
                              weight=weight_pd)

  modelos = {}

  for seed in semillas:
    logger.info(f"Construimos el modelo para la semilla : {seed}")

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

    best_iter = study.best_trial.user_attrs["best_iter"]
    new_params = study.best_trial.params
    new_params['n_estimators'] = best_iter
    
    params.update(new_params)
    model = lgb.train(params,train_data)

    modelos[seed] = model
    if is_test:
      model.save_model(os.path.join(modelos_directory,f"lgb_test_{seed}_{submission_number}.txt"))
    else:
      model.save_model(os.path.join(modelos_directory,f"lgb_predict_{seed}_{submission_number}.txt"))
  return modelos


###############################################################################3
# COMIENZA EL "MAIN"

logger.info(f"Config : {config}")

logger.info(f"Read Undersampled DataFrame : {os.path.join(BUCKETS,BUCKET_ORIGIN,"competencia_02_fe.csv")}")
df = pl.read_csv(os.path.join(BUCKETS,BUCKET_ORIGIN,"competencia_02_fe.csv"))
logger.info(f"Dataframe size : {df.shape}")

logger.info("Split Dataset")
logger.info(f"MES_TRAIN : {MES_TRAIN}")
logger.info(f"MES_VALIDACION : {MES_VALIDACION}")
logger.info(f"MES_TEST : {MES_TEST}")
logger.info(f"FINAL_PREDICT : {FINAL_PREDICT}")
logger.info(f"FINAL_TRAIN : {FINAL_TRAIN}")

logger.info("Split Clientes")
clientes_test = df.filter(pl.col('foto_mes') == MES_TEST)["numero_de_cliente"]
clientes_val = df.filter(pl.col('foto_mes') == MES_VALIDACION)["numero_de_cliente"]
clientes_predict = df.filter(pl.col('foto_mes') == FINAL_PREDICT)["numero_de_cliente"]

df = generate_clase_peso(df)

df_train = df.filter(pl.col('foto_mes').is_in(MES_TRAIN))
df_train = undersample_df(df_train, UNDERSAMPLE_FRACTION)

df = drop_columns(df)

df_test = df.filter(pl.col('foto_mes') == MES_TEST)
df_predict = df.filter(pl.col('foto_mes') == FINAL_PREDICT)
df_train_predict = df.filter(pl.col('foto_mes').is_in(FINAL_TRAIN))
df_val = df.filter(pl.col('foto_mes') == MES_VALIDACION)

df_train_clase_binaria_baja = df_train['clase_binaria']
df_test_clase_binaria_baja = df_test['clase_binaria']
df_predict_clase_binaria_baja = df_train_predict['clase_binaria']
df_val_clase_binaria = df_val['clase_binaria']

df_train_predict_weight = df_train_predict['clase_peso']
df_val_weight = df_val['clase_peso']
df_train_weight = df_train['clase_peso']

logger.info("Drop columns foto_mes, clase_binaria and clase_peso")

df_train = df_train.drop(['clase_binaria','clase_peso','foto_mes',"clase_ternaria"])
df_train_predict = df_train_predict.drop(['clase_binaria','clase_peso','foto_mes'])
if SUBMIT:
  df_train_predict = df_train_predict.drop(["clase_ternaria"])
  
df_val = df_val.drop(['clase_binaria','clase_peso','foto_mes',"clase_ternaria"])
df_test = df_test.drop(['clase_binaria','clase_peso','foto_mes',"clase_ternaria"])


# FUNCION OBJETIVO PARA OPTUNA
def objective(trial) -> float:

    logger.info(f"Begin Trial {trial.number}")
    num_leaves = trial.suggest_int('num_leaves', 8, 80)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.4)
    max_depth = trial.suggest_int("max_depth", 10, 100)
    min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 1, 1000)
    feature_fraction = trial.suggest_float('feature_fraction', 0.1, 1.0)
    max_bin = trial.suggest_int('max_bin', 255, 500)
    num_iterations = trial.suggest_int('num_iterations', 100, 500)

    opt_train_pd = df_train.to_pandas()
    opt_y_pd = df_train_clase_binaria_baja.to_pandas()
    opt_weight_pd = df_train_weight.to_pandas()

    logger.info(f"Opt Train Data : {len(opt_train_pd.columns)} , {opt_y_pd.shape} , {opt_weight_pd.shape}")
    train_data = lgb.Dataset(opt_train_pd,
                                label=opt_y_pd,
                                weight=opt_weight_pd.to_numpy())

    opt_X_val_pd = df_val.to_pandas()
    opt_y_val_pd = df_val_clase_binaria.to_pandas()
    weight_val_pd = df_val_weight.to_pandas()
    logger.info(f"Opt Val Data : {len(opt_X_val_pd.columns)} , {opt_y_val_pd.shape} , {weight_val_pd.shape}")
    val_data = lgb.Dataset(
          opt_X_val_pd,
          label=opt_y_val_pd,
          weight=weight_val_pd.to_numpy()
      )

    modelos = {}
    for s in SEMILLA:
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
        'seed': s,
        'verbose': -1,
        'num_iterations': num_iterations,
        }

      modelos[s] = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
      )
    
    optimization_predictions = build_predictions(clientes_val, modelos, df_val)
    ganancia_total,_ = ganancia_evaluator(optimization_predictions,df_val_clase_binaria)
    logger.info(f"Finished Trial {trial.number}: Ganancia = {ganancia_total}")
    return ganancia_total

# SE INTENTA RECUPERAR UN ESTUDIO O SE INICIA UNO NUEVO
storage_name = f"sqlite:////{os.path.join(BUCKETS, BUCKET_TARGET,STUDY_NAME)}.db"
study_name = STUDY_NAME

study = optuna.create_study(
    direction="maximize",
    study_name=study_name,
    storage=storage_name,
    load_if_exists=True,
)

# HAY UN FLAG EN EL CONFIG PARA EVITAR CORRER LA OPTIMIZACION SIEMPRE
if RUN_BAYESIAN_OPTIMIZATION:
  logger.info("Run Optimization")
  study.optimize(lambda trial: objective(trial), n_trials=50)


# QUIZAS NO SE QUIERE HACER TODA LA PRUEBA CON TEST ASI QUE SE HABILITA LA POSIBILIDAD DE SALTEAR ESTA PARTE
train_test_models = config["TRAIN_TEST_MODELS"]
test_models = {}

if train_test_models:
  for seed in SEMILLA:
    logger.info(f"Build or Load Test model for seed : {seed}")
    model_name = f"lgb_test_{seed}_{SUBMISSION_NUMBER}.txt"
    model_file_path = os.path.join(modelos_directory,model_name)
    if os.path.exists(model_file_path):
      logger.info(f"Cargamos el modelo de Test de la submission {SUBMISSION_NUMBER} para la semilla {seed}")
      booster = lgb.Booster(model_file=model_file_path)
      test_models[seed] = booster
      # SI NO SE ENCONTRO AL MENOS UNO DE LOS MODELOS ENTONCES HAY QUE ENTRENAR 
      train_test_models = False
    else:
      logger.info(f"El modelo de Test para la semilla {seed} no existe en {model_file_path}. Se entrenará.")

if train_test_models:
  test_models = build_and_save_models(study, SEMILLA, df_train, df_train_clase_binaria_baja, df_train_weight,is_test=True, run_bayesian_optimization=RUN_BAYESIAN_OPTIMIZATION)

# SIMILAR A PREDICCIÓN. EN ESTE CASO SE VA A HACER SIEMPRE
train_predict_models = True

predict_models = {}
for seed in SEMILLA:
  model_name = f"lgb_predict_{seed}_{SUBMISSION_NUMBER}.txt"
  model_file_path = os.path.join(modelos_directory,model_name)
  if os.path.exists(model_file_path):
    logger.info(f"Load Predict model for seed {seed}")
    modelo_anterior = lgb.Booster(model_file=model_file_path)
    predict_models[seed] = modelo_anterior
    train_predict_models = False
  else:
    logger.info(f"Predict model for seed {seed} does not exist. Will be trained.")

if train_predict_models:
  predict_models = build_and_save_models(study, SEMILLA,df_train_predict,df_predict_clase_binaria_baja, df_train_predict_weight, is_test=False, run_bayesian_optimization=RUN_BAYESIAN_OPTIMIZATION)

test_predictions = build_predictions(clientes_test, test_models, df_test)
ganancia, n_envios = ganancia_evaluator(test_predictions,df_val_clase_binaria)
logger.info(f"Ganancia en Test: {ganancia} con {n_envios} envios")

# PREPARAMOS EL DATASET DE PREDICCION PARA PASARLO POR EL MODELO
df_predict = df_predict.drop(['foto_mes'])
  
if not SUBMIT:
  df_predict_clase_binaria = df_predict["clase_binaria"]
  df_predict = df_predict.drop(['clase_peso', 'clase_binaria'])

comp_predictions = build_final_predictions(clientes_predict, predict_models, df_predict, n_envios)

# SI ESTAMOS EN ETAPA DE SUBMIT NO TIENE SENTIDO OBTENER LA GANANCIA 
if not SUBMIT:
  ganancia, n_envios = ganancia_evaluator(comp_predictions,df_predict_clase_binaria)
  logger.info(f"Ganancia en Prediccion de Experimento : {ganancia} con {n_envios} envios")
else:
  prediction_path = os.path.join(BUCKETS, BUCKET_TARGET, f"predictions_{SUBMISSION_NUMBER}.csv")
  logger.info(f"Build submission {prediction_path}")
  comp_predictions.write_csv(prediction_path)

logger.info(f"Program Ends")