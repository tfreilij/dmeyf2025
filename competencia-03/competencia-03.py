import datetime
import logging
import os
from typing import Dict, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl

from load_config import Config

# ============================================================================
# CONFIGURATION
# ============================================================================
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

# ============================================================================
# DIRECTORY SETUP
# ============================================================================
bucket_target = os.path.join(BUCKETS, BUCKET_TARGET)
modelos_directory = os.path.join(bucket_target, "modelos")
log_directory = os.path.join(bucket_target, "log")

os.makedirs(log_directory, exist_ok=True)
os.makedirs(modelos_directory, exist_ok=True)

# ============================================================================
# LOGGING SETUP
# ============================================================================
nombre_log = f"log_{STUDY_NAME}_{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}.log"
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

# ============================================================================
# VALIDATION
# ============================================================================
if BUCKET_ORIGIN == BUCKET_TARGET:
    raise RuntimeError("Bucket de Origen y Bucket de Destino no pueden ser iguales")

if BUCKET_TARGET == "b1":
    raise RuntimeError("Bucket Target no puede ser b1")

# ============================================================================
# DATA PREPROCESSING FUNCTIONS
# ============================================================================


def drop_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Drop columns that are not needed for training."""
    logger.info("Dropping columns")
    
    col_drops = [
        "mprestamos_personales",
        "cprestamos_personales"
    ]
    
    return df.drop(col_drops)


def generate_clase_binaria(df: pl.DataFrame) -> pl.DataFrame:
    """Binarize the target class (BAJA+2 -> 1, others -> 0)."""
    logger.info("Generating clase_binaria")
    
    return df.with_columns(
        pl.when(pl.col('clase_ternaria').is_in(['BAJA+2','BAJA+1']))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias('clase_binaria')
    )


def generate_clase_peso(df: pl.DataFrame) -> pl.DataFrame:
    """Generate class weights for training."""
    logger.info("Generating clase_peso")
    
    return (df
            .with_columns(pl.lit(1.0).alias('clase_peso'))
            .with_columns(
                pl.when(pl.col('clase_ternaria') == 'BAJA+2')
                .then(pl.lit(1.00002))
                .otherwise(pl.col('clase_peso'))
                .alias('clase_peso')
            )
            .with_columns(
                pl.when(pl.col('clase_ternaria') == 'BAJA+1')
                .then(pl.lit(1.00001))
                .otherwise(pl.col('clase_peso'))
                .alias('clase_peso')
            ))


def undersample_df(df: pl.DataFrame, fraction: float) -> pl.DataFrame:
    logger.info(f"Undersampling continuas with fraction: {fraction}, DF shape: {df.shape}")
    
    clientes_solo_continuas = (
        df.group_by("numero_de_cliente")
        .agg(
            n_bajas=pl.col("clase_ternaria")
            .is_in(["BAJA+1", "BAJA+2"])
            .sum()
        )
        .filter(pl.col("n_bajas") == 0)
    )
    
    clientes_solo_continuas_undersampled = clientes_solo_continuas.sample(
        fraction=1 - fraction, seed=1000
    )
    
    df_filtered = df.filter(
        ~pl.col('numero_de_cliente').is_in(
            clientes_solo_continuas_undersampled["numero_de_cliente"]
        )
    )
    
    logger.info(f"DF shape after undersampling: {df_filtered.shape}")
    return df_filtered


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================


def build_predictions(modelos: Dict, dataset: pl.DataFrame) -> pl.DataFrame:
    """Build averaged predictions from multiple models."""
    clientes = dataset["numero_de_cliente"]
    df_to_predict = dataset.drop(["numero_de_cliente"])
    dataset_np = df_to_predict.to_numpy()
    
    predicciones = {}
    for seed, model in modelos.items():
        if seed in SEMILLA:
            predictions = model.predict(dataset_np)
            predicciones[seed] = predictions
    
    mean_predictions = np.mean(list(predicciones.values()), axis=0)
    return pl.DataFrame({
        'numero_de_cliente': clientes,
        'Predicted': mean_predictions
    })


def build_final_predictions(
    predict_models: Dict,
    df_predict: pl.DataFrame,
    n_envios: int
) -> pl.DataFrame:
    """Build final predictions with top n_envios selected."""
    mean_predictions = build_predictions(predict_models, df_predict)
    sorted_mean_predictions = mean_predictions.sort('Predicted', descending=True)
    
    final_predictions = sorted_mean_predictions.with_columns(
        (pl.arange(0, sorted_mean_predictions.height) < n_envios)
        .cast(pl.Int8)
        .alias("Predicted")
    )
    
    return final_predictions.select(["numero_de_cliente", "Predicted"])


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================


def ganancia_optima_idealizada(df: pl.DataFrame, ternaria: pl.Series) -> float:
    """Calculate idealized optimal gain for comparison purposes."""
    df_ganancias = df.hstack(ternaria.to_frame())
    df_ganancias = df_ganancias.with_columns(
        pl.when(pl.col('clase_ternaria').is_in(["BAJA+2"]))
        .then(780000)
        .alias('ganancia_individual')
    )
    
    return df_ganancias['ganancia_individual'].sum()


def ganancia_evaluator(y_pred: pl.DataFrame, y_true: pl.DataFrame) -> Tuple[str, float, bool]:
    """Evaluate gain metric for LightGBM."""
    df_eval = y_pred.join(y_true, on="numero_de_cliente")
    df_ordenado = df_eval.sort("Predicted", descending=True)
    
    # Individual gain per row
    df_ordenado = df_ordenado.with_columns([
        pl.when(pl.col("clase_binaria") == 1)
        .then(GANANCIA_ACIERTO)
        .otherwise(-COSTO_ESTIMULO)
        .alias("ganancia_individual")
    ])
    
    # Cumulative gain
    df_ordenado = df_ordenado.with_columns([
        pl.col("ganancia_individual").cum_sum().alias("ganancia_acumulada")
    ])
    
    # Maximum gain
    ganancia_maxima = df_ordenado.select(pl.col("ganancia_acumulada").max()).item()
    
    # LightGBM expects: (metric_name, metric_value, is_higher_better)
    return 'ganancia_evaluator', float(ganancia_maxima), True


def cantidad_envios(y_pred: pl.DataFrame, y_true: pl.DataFrame) -> Tuple[float, int]:
    """Calculate maximum gain and corresponding number of sends."""
    df_eval = y_pred.join(y_true, on="numero_de_cliente")
    df_ordenado = df_eval.sort("Predicted", descending=True)
    
    # Individual gain per row
    df_ordenado = df_ordenado.with_columns([
        pl.when(pl.col("clase_binaria") == 1)
        .then(pl.lit(GANANCIA_ACIERTO).cast(pl.Float64))
        .otherwise(pl.lit(-COSTO_ESTIMULO).cast(pl.Float64))
        .alias("ganancia_individual")
    ])
    
    # Cumulative gain
    df_ordenado = df_ordenado.with_columns([
        pl.col("ganancia_individual").cum_sum().alias("ganancia_acumulada")
    ])
    
    # Maximum gain
    ganancia_maxima = df_ordenado.select(pl.col("ganancia_acumulada").max()).item()
    
    # Find index of maximum cumulative gain
    idx_max_ganancia = df_ordenado["ganancia_acumulada"].arg_max()
    cantidad_envios_real = idx_max_ganancia + 1
    
    return float(ganancia_maxima), cantidad_envios_real


def lgb_gan_eval(y_pred, data: lgb.Dataset) -> Tuple[str, float, bool]:
    """Custom gain evaluation function for LightGBM."""
    weight = data.get_weight()
    
    ganancia = (
        np.where(weight == 1.00002, GANANCIA_ACIERTO, 0) -
        np.where(weight < 1.00002, COSTO_ESTIMULO, 0)
    )
    
    ganancia = ganancia[np.argsort(y_pred)[::-1]]
    ganancia = np.cumsum(ganancia)
    
    return "gan_eval", float(np.max(ganancia)), True


# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================


def build_and_save_or_load_models(
    study: optuna.Study,
    semillas: list,
    train_dataset: pl.DataFrame,
    y_target: pl.DataFrame,
    undersampling_fraction: float
) -> Dict:
    """Build, save, or load models for prediction."""
    if undersampling_fraction is None:
        raise RuntimeError(
            f"Undersampling Fraction {undersampling_fraction} no puede ser None"
        )
    
    modelos = {}
    
    # Check if all models exist
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
        logger.info("Training predict models (some models were missing)")
        
        train_dataset_pd = train_dataset.to_numpy()
        y_target_np = y_target["clase_binaria"].to_numpy()
        weight_np = y_target["clase_peso"].to_numpy()
        
        train_data = lgb.Dataset(
            train_dataset_pd,
            label=y_target_np,
            weight=weight_np
        )
        
        if len(study.trials) == 0:
            raise RuntimeError("No trials found in study. Run optimization first.")
        
        best_params = study.best_trial.params.copy()
        best_iter = study.best_trial.user_attrs.get("best_iteration", 110)
        
        # Adjust min_data_in_leaf for undersampling
        best_params["min_data_in_leaf"] = int(
            best_params["min_data_in_leaf"] * 100 / undersampling_fraction
        )
        
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
                **best_params
            }
            
            model = lgb.train(params, train_data, num_boost_round=best_iter)
            modelos[seed] = model
            
            model_path = os.path.join(modelos_directory, f"lgb_predict_{seed}.txt")
            model.save_model(model_path)
    
    return modelos


# ============================================================================
# OPTUNA OPTIMIZATION
# ============================================================================


def create_objective_function(
    df_train: pl.DataFrame,
    df_train_with_target: pl.DataFrame,
    df_val: pl.DataFrame,
    df_val_with_target: pl.DataFrame,
    df_test: pl.DataFrame,
    df_test_with_target: pl.DataFrame
):
    """Create objective function with access to dataframes."""
    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function for hyperparameter optimization."""
        logger.info(f"Begin Trial {trial.number}")
        
        # Suggest hyperparameters
        num_leaves = trial.suggest_int('num_leaves', 8, 80)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.4)
        max_depth = trial.suggest_int("max_depth", 10, 100)
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 1, 1000)
        feature_fraction = trial.suggest_float('feature_fraction', 0.1, 1.0)
        max_bin = trial.suggest_int('max_bin', 255, 500)
        num_iterations = trial.suggest_int('num_iterations', 100, 500)
        
        # Prepare training data
        opt_train_pd = df_train.to_numpy()
        opt_y_pd = df_train_with_target["clase_binaria"].to_numpy()
        opt_weight_pd = df_train_with_target["clase_peso"].to_numpy()
        
        train_data = lgb.Dataset(
            opt_train_pd,
            label=opt_y_pd,
            weight=opt_weight_pd
        )
        
        # Prepare validation data
        opt_X_val_pd = df_val.to_numpy()
        opt_y_val_pd = df_val_with_target["clase_binaria"].to_numpy()
        weight_val_pd = df_val_with_target["clase_peso"].to_numpy()
        
        logger.info(f"VAL DATA: {opt_X_val_pd.shape}")
        
        val_data = lgb.Dataset(
            opt_X_val_pd,
            label=opt_y_val_pd,
            weight=weight_val_pd
        )
        
        # Train models with different seeds
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
                feval=lgb_gan_eval,
                callbacks=[
                    lgb.early_stopping(
                        stopping_rounds=int(50 + 5 / params['learning_rate']),
                        verbose=False
                    )
                ]
            )
            modelos[s] = model
        
        # Evaluate on test set
        optimization_predictions = build_predictions(modelos, df_test)
        ganancia_total, n_envios_test = cantidad_envios(
            optimization_predictions,
            df_test_with_target
        )
        
        logger.info(
            f"Finished Trial {trial.number}: Ganancia = {ganancia_total}, "
            f"n_envios = {n_envios_test}"
        )
        
        # Store trial attributes
        trial.set_user_attr("ganancia_maxima", ganancia_total)
        trial.set_user_attr("n_envios", n_envios_test)
        trial.set_user_attr("best_iteration", model.best_iteration)
        
        return ganancia_total
    
    return objective


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Main execution function."""
    logger.info(f"Config: {config}")
    
    # Load data
    data_path = os.path.join(BUCKETS, BUCKET_ORIGIN, "competencia_02_fe.csv")
    logger.info(f"Reading DataFrame: {data_path}")
    df_fe = pl.read_csv(data_path)
    logger.info(f"DataFrame size: {df_fe.shape}")
    
    # Data preprocessing
    logger.info("Preprocessing data")
    df_fe = df_fe.sort(by=["numero_de_cliente", "foto_mes"], descending=[False, False])
    df = generate_clase_peso(df_fe)
    df = generate_clase_binaria(df)
    df = drop_columns(df)
    
    # Split datasets
    logger.info("Splitting dataset")
    logger.info(f"MES_TRAIN: {MES_TRAIN}")
    logger.info(f"MES_VALIDACION: {MES_VALIDACION}")
    logger.info(f"MES_TEST: {MES_TEST}")
    logger.info(f"FINAL_PREDICT: {FINAL_PREDICT}")
    logger.info(f"FINAL_TRAIN: {FINAL_TRAIN}")
    
    df_train = df.filter(pl.col('foto_mes').is_in(MES_TRAIN))
    df_train = undersample_df(df_train, UNDERSAMPLE_FRACTION)
    
    df_test = df.filter(pl.col('foto_mes') == MES_TEST)
    df_predict = df.filter(pl.col('foto_mes') == FINAL_PREDICT)
    df_train_predict = df.filter(pl.col('foto_mes').is_in(FINAL_TRAIN))
    df_val = df.filter(pl.col('foto_mes') == MES_VALIDACION)
    
    # Extract targets before dropping columns
    df_train_with_target = df_train.select([
        'numero_de_cliente', 'clase_binaria', 'clase_peso'
    ])
    df_val_with_target = df_val.select([
        'numero_de_cliente', 'clase_binaria', 'clase_peso'
    ])
    df_test_with_target = df_test.select([
        'numero_de_cliente', 'clase_binaria', 'clase_peso', 'clase_ternaria'
    ])
    df_train_predict_with_target = df_train_predict.select([
        'numero_de_cliente', 'clase_binaria', 'clase_peso'
    ])
    
    if IS_EXPERIMENTO:
        df_predict_with_target = df_predict.select([
            'numero_de_cliente', 'clase_binaria', 'clase_peso', 'clase_ternaria'
        ])
    else:
        df_predict_with_target = df_predict.select(['numero_de_cliente'])
    
    # Drop target columns from feature datasets
    cols_to_drop = ['numero_de_cliente', 'clase_binaria', 'clase_peso', 'foto_mes', 'clase_ternaria']
    
    df_train = df_train.drop(cols_to_drop)
    df_train_predict = df_train_predict.drop(cols_to_drop)
    df_val = df_val.drop(['clase_binaria', 'clase_peso', 'foto_mes', 'clase_ternaria'])
    df_test = df_test.drop(['clase_binaria', 'clase_peso', 'foto_mes', 'clase_ternaria'])
    
    if IS_EXPERIMENTO:
        df_predict = df_predict.drop(['clase_binaria', 'clase_peso', 'foto_mes', 'clase_ternaria'])
    
    # Log class distributions
    logger.info(f"Train target distribution:\n{df_train_with_target['clase_binaria'].value_counts()}")
    logger.info(f"Test target distribution:\n{df_test_with_target['clase_binaria'].value_counts()}")
    
    # Setup Optuna study
    storage_name = f"sqlite:////{os.path.join(BUCKETS, BUCKET_TARGET, STUDY_NAME)}.db"
    study = optuna.create_study(
        direction="maximize",
        study_name=STUDY_NAME,
        storage=storage_name,
        load_if_exists=True,
    )
    
    # Create objective function with access to dataframes
    objective_func = create_objective_function(
        df_train,
        df_train_with_target,
        df_val,
        df_val_with_target,
        df_test,
        df_test_with_target
    )
    
    # Run optimization if enabled
    if RUN_BAYESIAN_OPTIMIZATION:
        logger.info(f"Running optimization with {N_TRIALS} trials")
        study.optimize(objective_func, n_trials=N_TRIALS)
    
    # Build or load prediction models
    predict_models = build_and_save_or_load_models(
        study,
        SEMILLA,
        df_train_predict,
        df_train_predict_with_target,
        undersampling_fraction=UNDERSAMPLE_FRACTION
    )
    
    # Get best n_envios from study
    best_n_envios = study.best_trial.user_attrs["n_envios"]
    logger.info(f"Best n_envios: {best_n_envios}")
    
    # Make final predictions
    if IS_EXPERIMENTO:
        if "clase_ternaria" in df_predict_with_target.columns:
            df_predict_ternaria = df_predict_with_target["clase_ternaria"]
            ganancia_optima = ganancia_optima_idealizada(df_predict, df_predict_ternaria)
            logger.info(f"Ganancia 'optima' en Prediccion usada como pruebas: {ganancia_optima}")
        
        # Ensure target columns are dropped from df_predict
        cols_to_drop = [
            col for col in ['clase_binaria', 'clase_peso', 'clase_ternaria']
            if col in df_predict.columns
        ]
        if cols_to_drop:
            logger.info(f"Dropping columns from df_predict: {cols_to_drop}")
            df_predict = df_predict.drop(cols_to_drop)
        
        comp_predictions = build_final_predictions(predict_models, df_predict, best_n_envios)
        logger.info(f"Prediction distribution:\n{comp_predictions['Predicted'].value_counts()}")
        
        ganancia, n_envios_final = cantidad_envios(comp_predictions, df_predict_with_target)
        logger.info(
            f"Ganancia en Prediccion de Experimento: {ganancia} con {n_envios_final} envios"
        )
    else:
        # Ensure target columns are dropped from df_predict
        cols_to_drop = [
            col for col in ['clase_binaria', 'clase_peso', 'clase_ternaria']
            if col in df_predict.columns
        ]
        if cols_to_drop:
            logger.info(f"Dropping columns from df_predict: {cols_to_drop}")
            df_predict = df_predict.drop(cols_to_drop)
        
        prediction_path = os.path.join(BUCKETS, BUCKET_TARGET, "predictions.csv")
        logger.info(f"Building submission: {prediction_path}")
        
        comp_predictions = build_final_predictions(predict_models, df_predict, best_n_envios)
        comp_predictions.write_csv(prediction_path)
    
    logger.info("Program completed successfully")


if __name__ == "__main__":
    main()
