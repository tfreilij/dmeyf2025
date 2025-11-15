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

## VARIABLES DE ENTORNO Y CONFIGURACION
config = Config()
MES_TRAIN = config["MES_TRAIN"]
BUCKETS = config["BUCKETS"]
BUCKET_ORIGIN = config["BUCKET_ORIGIN"]
BUCKET_TARGET = config["BUCKET_TARGET"]
IS_EXPERIMENTO = config["IS_EXPERIMENTO"]
MES_VALIDACION = config["MES_VALIDACION"]
STUDY_NAME = config["STUDY_NAME"]
GANANCIA_ACIERTO = config["GANANCIA_ACIERTO"]
COSTO_ESTIMULO = config["COSTO_ESTIMULO"]
FINAL_PREDICT = config["FINAL_PREDICT"]
MES_TEST = config["MES_TEST"]
FINAL_TRAIN = config["FINAL_TRAIN"]
SEMILLA = config["SEMILLA"]
UNDERSAMPLE_FRACTION = config["UNDERSAMPLING_FRACTION"]
RUN_BAYESIAN_OPTIMIZATION = config["RUN_BAYESIAN_OPTIMIZATION"]
N_TRIALS = config["N_TRIALS"]

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

logger = logging.getLogger(__name__)


if BUCKET_ORIGIN == BUCKET_TARGET:
  raise RuntimeError("Bucket de Origen y Bucket de Destino no pueden ser iguales")

if BUCKET_TARGET == "b1":
  raise RuntimeError("Bucket Target no puede ser b1")

## DROPEAR COLUMNAS QUE NO VAN PARA ENTRENAMIENTO
def drop_columns(df : pl.DataFrame):
    logger.info("Drop columns")

    col_drops = ["Visa_Finiciomora","Visa_Finiciomora",
          "Visa_fultimo_cierre", "Master_fultimo_cierre",
          "Visa_Fvencimiento", "Master_Fvencimiento",'tmobile_app','mprestamos_personales'
      ]

    df = df.drop(col_drops)
    return df

## SE ARMAN LAS PREDICCIONES PROMEDIADAS
def build_predictions(modelos, dataset : pl.DataFrame) -> pl.DataFrame:
  predicciones = {}
  clientes = dataset["numero_de_cliente"]
  df_to_predict = dataset.drop(["numero_de_cliente"])

  # Convert the dataset to a NumPy array for LightGBM prediction
  dataset_np = df_to_predict.to_numpy()

  for seed,model in modelos.items():
    if seed in SEMILLA:
      predictions = model.predict(dataset_np)
      predicciones[seed] = predictions

  mean_predictions = np.mean(list(predicciones.values()), axis=0)
  return pl.DataFrame({'numero_de_cliente': clientes, 'Predicted': mean_predictions})

## FUNCION AUXILIAR QUE ARMO PARA TENER UNA MINIMA COMPARACION CON LO "ESPERADO"
def ganancia_optima_idealizada(df :pl.DataFrame, ternaria : pl.Series) -> float:

  df_ganancias = df.hstack(ternaria.to_frame())
  df_ganancias = df_ganancias.with_columns(
      pl.when(pl.col('clase_ternaria').is_in(["BAJA+2"]))
        .then(780000)
        .alias('ganancia_individual')
  )

  ganancia = df_ganancias['ganancia_individual'].sum()
  return ganancia

## SE ARMAN LAS PREDICCIONES PARA EL TARGET
def build_final_predictions(predict_models, df_predict, n_envios):
  mean_predictions = build_predictions(predict_models, df_predict)
  sorted_mean_predictions = mean_predictions.sort('Predicted', descending=True)
  final_predictions = sorted_mean_predictions.with_columns(
        (pl.arange(0, sorted_mean_predictions.height) < n_envios)
        .cast(pl.Int8)
        .alias("Predicted")
    )
  
  return final_predictions.select(["numero_de_cliente", "Predicted"])

## SE APLICAR UNDERSAMPLING. SÓLO DEBERÍA USARSE EN EL DF_TRAIN
def undersample_df(df: pl.DataFrame, fraction) -> pl.DataFrame:
  logger.info(f"Undersampling Continuas with fraction : {fraction} , DF shape : {df.shape}")
  
  clientes_solo_continuas = df.group_by("numero_de_cliente").agg(n_bajas=pl.col("clase_binaria").sum()).filter(pl.col("n_bajas") == 0)
  clientes_solo_continuas_undersampled = clientes_solo_continuas.sample(fraction=1-fraction, seed=1000)
  df = df.filter(~pl.col('numero_de_cliente').is_in(clientes_solo_continuas_undersampled["numero_de_cliente"]))
  logger.info(f"DF shape after undersampling: {df.shape}")
  return df

## SE ORDENAN DE MAYOR A MENOR LAS PROBABILIDADES Y SE BUSCA LA MAXIMA GANANCIA JUNTO A LA CANTIDAD DE ENVIOS CORRESPONDIENTES
def ganancia_evaluator(y_pred : pl.DataFrame, y_true : pl.DataFrame) -> float:

    df_eval = y_pred.join(y_true, on="numero_de_cliente")

    df_ordenado = df_eval.sort("Predicted", descending=True)

    # Ganancia individual por fila
    df_ordenado = df_ordenado.with_columns([
        pl.when(pl.col("clase_binaria") == 1)
          .then(GANANCIA_ACIERTO)
          .otherwise(-COSTO_ESTIMULO)
          .alias("ganancia_individual")
    ])

    # Ganancia acumulada
    df_ordenado = df_ordenado.with_columns([
        pl.col("ganancia_individual").cum_sum().alias("ganancia_acumulada")
    ])

    # Obtener ganancia maxima
    ganancia_maxima = df_ordenado.select(pl.col("ganancia_acumulada").max()).item()

    # LightGBM espera: (nombre_metric, valor_metric, is_higher_better)
    return 'ganancia_evaluator', float(ganancia_maxima), True

def cantidad_envios(y_pred : pl.DataFrame, y_true : pl.DataFrame) -> float:
    df_eval = y_pred.join(y_true, on="numero_de_cliente")

    df_ordenado = df_eval.sort("Predicted", descending=True)

    # Ganancia individual por fila, cast to Float64 to prevent potential overflow
    df_ordenado = df_ordenado.with_columns([
        pl.when(pl.col("clase_binaria") == 1)
          .then(pl.lit(GANANCIA_ACIERTO).cast(pl.Float64))
          .otherwise(pl.lit(-COSTO_ESTIMULO).cast(pl.Float64))
          .alias("ganancia_individual")
    ])

    # Ganancia acumulada
    df_ordenado = df_ordenado.with_columns([
        pl.col("ganancia_individual").cum_sum().alias("ganancia_acumulada")
    ])

    # Obtener ganancia maxima
    ganancia_maxima = df_ordenado.select(pl.col("ganancia_acumulada").max()).item()

    # Find the index of the first occurrence of the maximum cumulative gain
    idx_max_ganancia = df_ordenado["ganancia_acumulada"].arg_max()

    # The number of sends is the index + 1 (since index is 0-based)
    cantidad_envios_real = idx_max_ganancia + 1

    return float(ganancia_maxima), cantidad_envios_real

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

def lgb_gan_eval(y_pred, data: lgb.Dataset):
  weight = data.get_weight()

  ganancia = np.where(weight == 1.00002, GANANCIA_ACIERTO, 0) - np.where(
      weight < 1.00002, COSTO_ESTIMULO, 0
  )

  ganancia = ganancia[np.argsort(y_pred)[::-1]]
  ganancia = np.cumsum(ganancia)

  return "gan_eval", float(np.max(ganancia)), True
## SE BINARIZA LA CLASE OBJETIVO.
def generate_clase_binaria(df : pl.DataFrame):

    df = df.with_columns(pl.lit(0).alias('clase_binaria'))

    df = df.with_columns(
        pl.when(pl.col('clase_ternaria').is_in(['BAJA+2'])).then(pl.lit(1)).otherwise(pl.lit(0)).alias('clase_binaria')
    )

    return df

## SE ARMA EL MODELO Y DE SER POSIBLE SE PERSISTE PARA PODER USARLO PARA OTRA PREDICCIÓN.
def build_and_save_or_load_models(study, semillas : list, train_dataset : pl.DataFrame, y_target : pl.DataFrame, undersampling_fraction) -> dict:

  if undersampling_fraction == None:
    raise RuntimeError(f"Undersampling Fraction {undersampling_fraction} no puede ser None")

  modelos = {} # Initialize modelos dictionary

  # Check if all models exist, if so, load them
  all_models_exist = True
  for seed in SEMILLA:
    model_name = f"lgb_predict_{seed}.txt"
    model_file_path = os.path.join(modelos_directory, model_name)
    if not os.path.exists(model_file_path):
      all_models_exist = False
      break

  if all_models_exist:
    logger.info("All predict models exist. Loading them.")
    for seed in SEMILLA:
        model_name = f"lgb_predict_{seed}.txt"
        model_file_path = os.path.join(modelos_directory, model_name)
        modelos[seed] = lgb.Booster(model_file=model_file_path)
  else:
    logger.info("Train Predict Models (some models were missing or not all present)")
    train_dataset_pd = train_dataset.to_pandas()
    y_target_np = y_target["clase_binaria"].to_numpy()
    weight_np = y_target["clase_peso"].to_numpy()

    train_data = lgb.Dataset(train_dataset_pd,
                                label=y_target_np,
                                weight=weight_np)

    if len(study.trials) == 0:
      raise RuntimeError("No trials found in study. Run optimization first.")

    best_params = study.best_trial.params.copy()
    # Retrieve best_iteration from user_attrs, default to 110 if not found
    best_iter = study.best_trial.user_attrs.get("best_iteration", 110)

    # Adjust min_data_in_leaf for undersampling
    best_params["min_data_in_leaf"] = int(best_params["min_data_in_leaf"] * 100 / undersampling_fraction)

    for seed in semillas:
      params = {
              'objective': 'binary',
                'metric': 'custom',
                'boosting_type': 'rf',
                'first_metric_only': True,
                'boost_from_average': True,
                'feature_pre_filter': False,
                'seed': seed,
                'verbose': -1,
                **best_params # Apply best parameters from Optuna
          }

      model = lgb.train(params, train_data, num_boost_round=best_iter)

      modelos[seed] = model
      model.save_model(os.path.join(modelos_directory,f"lgb_predict_{seed}.txt"))

  return modelos

###############################################################################3
# COMIENZA EL "MAIN"

logger.info(f"Config : {config}")

logger.info(f"Read DataFrame : {os.path.join(BUCKETS,BUCKET_ORIGIN,"competencia_02_fe.csv")}")
df_fe = pl.read_csv(os.path.join(BUCKETS,BUCKET_ORIGIN,"competencia_02_fe.csv"))
logger.info(f"Dataframe size : {df_fe.shape}")

df_crudo = df_fe.sort(by=["numero_de_cliente", "foto_mes"], descending=[False, False])

logger.info("Split Dataset")
logger.info(f"MES_TRAIN : {MES_TRAIN}")
logger.info(f"MES_VALIDACION : {MES_VALIDACION}")
logger.info(f"MES_TEST : {MES_TEST}")
logger.info(f"FINAL_PREDICT : {FINAL_PREDICT}")
logger.info(f"FINAL_TRAIN : {FINAL_TRAIN}")


df_peso = generate_clase_peso(df_fe)
df_binaria = generate_clase_binaria(df_peso)
df = drop_columns(df_binaria)

logger.info(f"Resumen de clase ternaria {df['clase_ternaria'].value_counts()}")

logger.info(f"Resumen de clase binaria {df['clase_binaria'].value_counts()}")

df_train = df.filter(pl.col('foto_mes').is_in(MES_TRAIN))
df_train = undersample_df(df_train, UNDERSAMPLE_FRACTION)

logger.info(df_train["clase_binaria"].value_counts())

df_test = df.filter(pl.col('foto_mes') == MES_TEST)
df_predict = df.filter(pl.col('foto_mes') == FINAL_PREDICT)
df_train_predict = df.filter(pl.col('foto_mes').is_in(FINAL_TRAIN))
df_val = df.filter(pl.col('foto_mes') == MES_VALIDACION)

# Crear DataFrames de alineación con numero_de_cliente y clase_binaria antes de cualquier drop
df_train_with_target = df_train.select(['numero_de_cliente', 'clase_binaria','clase_peso'])
df_val_with_target = df_val.select(['numero_de_cliente', 'clase_binaria','clase_peso'])
df_test_with_target = df_test.select(['numero_de_cliente', 'clase_binaria','clase_peso'])
df_train_predict_with_target = df_train_predict.select(['numero_de_cliente', 'clase_binaria','clase_peso'])
df_predict_with_target = df_predict.select(['numero_de_cliente'])
if IS_EXPERIMENTO:
  df_predict_with_target = df_predict.select(['numero_de_cliente', 'clase_binaria','clase_peso','clase_ternaria'])

df_train = df_train.drop(['clase_binaria','clase_peso','foto_mes',"clase_ternaria"])
df_train_predict = df_train_predict.drop(['clase_binaria','clase_peso','foto_mes',"clase_ternaria"])
df_val = df_val.drop(['clase_binaria','clase_peso','foto_mes',"clase_ternaria"])
df_test_ternaria = df_test["clase_ternaria"]
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

    opt_train_pd = df_train.to_numpy()
    opt_y_pd = df_train_with_target["clase_binaria"].to_numpy()
    opt_weight_pd = df_train_with_target["clase_peso"].to_numpy()

    train_data = lgb.Dataset(opt_train_pd,
                                label=opt_y_pd,
                                weight=opt_weight_pd)


    opt_X_val_pd = df_val.to_numpy()
    opt_y_val_pd = df_val_with_target["clase_binaria"].to_numpy()
    weight_val_pd = df_val_with_target["clase_peso"].to_numpy()
    logger.info(f"VAL DATA : {opt_X_val_pd.shape}")
    val_data = lgb.Dataset(
        opt_X_val_pd,
        label=opt_y_val_pd,
        weight=weight_val_pd
    )

    modelos = {}
    for s in SEMILLA:
      params = {
        'objective': 'binary',
        'metric': 'custom',
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



      model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        feval = lgb_gan_eval,
        callbacks=[
                lgb.early_stopping(stopping_rounds=int(50 + 5 / params['learning_rate']), verbose=False)
            ]

      )
      modelos[s] = model

    optimization_predictions = build_predictions(modelos, df_test)
    print(f"test target: {df_test_with_target["clase_binaria"].value_counts()}")
    ganancia_total, n_envios_test = cantidad_envios(optimization_predictions, df_test_with_target)
    print(f"Resultado: {ganancia_total}, {n_envios_test}")
    logger.info(f"Finished Trial {trial.number}: Ganancia = {ganancia_total}")

    trial.set_user_attr("ganancia_maxima", ganancia_total)
    trial.set_user_attr("n_envios", n_envios_test)
    trial.set_user_attr("best_iteration", model.best_iteration)

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
  logger.info(f"Run Optimization with {N_TRIALS}")
  study.optimize(lambda trial: objective(trial), n_trials=N_TRIALS)


predict_models = build_and_save_or_load_models(study, SEMILLA,df_train_predict, df_train_predict_with_target, undersampling_fraction=UNDERSAMPLE_FRACTION)

# Retrieve best n_envios from the study's best trial
best_n_envios = study.best_trial.user_attrs["n_envios"]
print(best_n_envios)
if IS_EXPERIMENTO:
  if "clase_ternaria" in df_predict_with_target.columns:
    df_predict_ternaria = df_predict_with_target["clase_ternaria"]
    logger.info(f"Ganancia 'optima' en Prediccion usada como pruebas: {ganancia_optima_idealizada(df_predict, df_predict_ternaria)}")

  cols_to_drop = [col for col in ['clase_binaria','clase_peso',"clase_ternaria"] if col in df_predict.columns]
  if cols_to_drop:
    logger.info(f"Dropping columns from df_predict: {cols_to_drop}")
    df_predict = df_predict.drop(cols_to_drop)

  comp_predictions = build_final_predictions(predict_models, df_predict, best_n_envios)
  print(comp_predictions["Predicted"].value_counts())
  ganancia, n_envios_final = cantidad_envios(comp_predictions, df_predict_with_target)
  #logger.info(f"Ganancia en Prediccion de Experimento : {ganancia} con {n_envios_final} envios")
  print(f"Ganancia en Prediccion de Experimento : {ganancia} con {n_envios_final} envios")
else:

  cols_to_drop = [col for col in ['clase_binaria','clase_peso',"clase_ternaria"] if col in df_predict.columns]
  if cols_to_drop:
    logger.info(f"Dropping columns from df_predict: {cols_to_drop}")
    df_predict = df_predict.drop(cols_to_drop)

  prediction_path = os.path.join(BUCKETS, BUCKET_TARGET, f"predictions.csv")
  logger.info(f"Build submission {prediction_path}")
  comp_predictions = build_final_predictions(predict_models, df_predict, best_n_envios)
  comp_predictions.write_csv(prediction_path)

logger.info(f"Program Ends")

logger.info(f"Program Ends")