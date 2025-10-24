# src/optimization.py (actualizar)
import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
from .config import *
from .gain_function import calcular_ganancia, ganancia_lgb_binary, ganancia_evaluator
from .loader import convertir_clase_ternaria_a_target

logger = logging.getLogger(__name__)

def objetivo_ganancia(trial: optuna.trial.Trial, df: pd.DataFrame, undersampling: float = 1) -> float:
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
    params = {
        'objective': 'binary',
        'metric': 'None',  # Usamos nuestra métrica personalizada

	#completar a gusto!!!!!!!


        'min_gain_to_split': 0.0,  # Permitir splits con ganancia mínima
        'verbose': -1,  # Reducir verbosidad
        'verbosity': -1,  # Silenciar mensajes adicionales
        'silent': True,  # Modo silencioso
        'bin': 31,
        'random_state': SEMILLA[0],  # Desde configuración YAML
    }
  
    if isinstance(MES_TRAIN, list):
        df_train = df[df['foto_mes'].isin(MES_TRAIN)]
    else:
        df_train = df[df['foto_mes'] == MES_TRAIN]
  
    df_val = df[df['foto_mes'] == MES_VALIDACION]
  
    #Convierto a binaria la clase ternaria, 
    # para entrenar el modelo Baja+1 y Baja+2 == 1
    # y calcular la ganancia de validacion Baja+2 solamente en 1
    df_train = convertir_clase_ternaria_a_target(df_train, baja_2_1=True)
    df_val = convertir_clase_ternaria_a_target(df_val, baja_2_1=False)
    df_train['clase_ternaria'] = df_train['clase_ternaria'].astype(np.int8)
    df_val['clase_ternaria'] = df_val['clase_ternaria'].astype(np.int8)

    # Usar target (clase_ternaria ya convertida a binaria)
  
    # Features: usar todas las columnas excepto target
  
    # Entrenar modelo con función de ganancia personalizada

  
    #####
    #ESTO NO ES OPTIMO!
    #####
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        feval=ganancia_evaluator,  # Función de ganancia personalizada
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
  
    # Predecir y calcular ganancia
    y_pred_proba = model.predict(X_val)
  
    #COMO OBSERVARON USO LA MISMA FUNCION DE GANANCIA PARA ENTRENAR Y VALIDAR, CAMBIEN A GUSTO!!!!!
    _, ganancia_total, _ = ganancia_evaluator(y_pred_proba, val_data)

    # Guardar cada iteración en JSON
    guardar_iteracion(trial, ganancia_total)
  
    logger.info(f"Trial {trial.number}: Ganancia = {ganancia_total:,.0f}")
  
    return ganancia_total


def crear_o_cargar_estudio(study_name: str = None, semilla: int = None) -> optuna.Study:
    """
    Crea un nuevo estudio de Optuna o carga uno existente desde SQLite.
  
    Args:
        study_name: Nombre del estudio (si es None, usa STUDY_NAME del config)
        semilla: Semilla para reproducibilidad
  
    Returns:
        optuna.Study: Estudio de Optuna (nuevo o cargado)
    """
    study_name = STUDY_NAME
  
    if semilla is None:
        semilla = SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA
  
    # Crear carpeta para bases de datos si no existe
    path_db = os.path.join(BUCKET_NAME, "optuna_db")
    os.makedirs(path_db, exist_ok=True)
  
    # Ruta completa de la base de datos
    db_file = os.path.join(path_db, f"{study_name}.db")
    storage = f"sqlite:///{db_file}"
  
    # Verificar si existe un estudio previo
    if os.path.exists(db_file):
        logger.info(f"⚡ Base de datos encontrada: {db_file}")
        logger.info(f"🔄 Cargando estudio existente: {study_name}")
  
        try:
            #PRESTAR ATENCION Y RAZONAR!!!
            study = optuna.load_study(study_name=study_name, storage=storage)
            n_trials_previos = len(study.trials)
  
            logger.info(f"✅ Estudio cargado exitosamente")
            logger.info(f"📊 Trials previos: {n_trials_previos}")
  
            if n_trials_previos > 0:
                logger.info(f"🏆 Mejor ganancia hasta ahora: {study.best_value:,.0f}")
  
            return study
  
        except Exception as e:
            logger.warning(f"⚠️ No se pudo cargar el estudio: {e}")
            logger.info(f"🆕 Creando nuevo estudio...")
    else:
        logger.info(f"🆕 No se encontró base de datos previa")
        logger.info(f"📁 Creando nueva base de datos: {db_file}")
  
    # Crear nuevo estudio
    study = optuna.create_study(
        #COMPLETAR
    )
  
    logger.info(f"✅ Nuevo estudio creado: {study_name}")
    logger.info(f"💾 Storage: {storage}")
  
    return study



def optimizar(df: pd.DataFrame, n_trials: int, study_name: str = None, undersampling: float = 0.01) -> optuna.Study:
    """
    Args:
        df: DataFrame con datos
        n_trials: Número de trials a ejecutar
        study_name: Nombre del estudio (si es None, usa el de config.yaml)
        undersampling: Undersampling para entrenamiento
  
    Description:
       Ejecuta optimización bayesiana de hiperparámetros usando configuración YAML.
       Guarda cada iteración en un archivo JSON separado. 
       Pasos:
        1. Crear estudio de Optuna
        2. Ejecutar optimización
        3. Retornar estudio

    Returns:
        optuna.Study: Estudio de Optuna con resultados
    """

    study_name = STUDY_NAME

    logger.info(f"Iniciando optimización con {n_trials} trials")
    logger.info(f"Configuración: TRAIN={MES_TRAIN}, VALID={MES_VALIDACION}, SEMILLA={SEMILLA}")
  
    # Crear o cargar estudio desde DuckDB
    study = crear_o_cargar_estudio(study_name, SEMILLA)

    # Calcular cuántos trials faltan
    trials_previos = len(study.trials)
    trials_a_ejecutar = max(0, n_trials - trials_previos)
  
    if trials_previos > 0:
        logger.info(f"🔄 Retomando desde trial {trials_previos}")
        logger.info(f"📝 Trials a ejecutar: {trials_a_ejecutar} (total objetivo: {n_trials})")
    else:
        logger.info(f"🆕 Nueva optimización: {n_trials} trials")
  
    # Ejecutar optimización
    if trials_a_ejecutar > 0:
        ##LO UNICO IMPORTANTE DEL METODO Y EL study CLARO
        study.optimize(lambda trial: objetivo_ganancia(trial, df, undersampling), n_trials=trials_a_ejecutar)
        logger.info(f"🏆 Mejor ganancia: {study.best_value:,.0f}")
        logger.info(f"Mejores parámetros: {study.best_params}")
    else:
        logger.info(f"✅ Ya se completaron {n_trials} trials")
  
    return study

def aplicar_undersampling(df: pd.DataFrame, ratio: float, random_state: int = None) -> pd.DataFrame:
    pass

# loader.py
def convertir_clase_ternaria_a_target(df: pd.DataFrame, baja_2_1=True) -> pd.DataFrame:
    """
    Convierte clase_ternaria a target binario reemplazando en el mismo atributo:
    - CONTINUA = 0
    y segun los argumentos baja_2_1
    baja_2_1 = true entonces: BAJA+1 y BAJA+2 = 1
    baja_2_1 = false entonces: BAJA+1 = 0 y BAJA+2 = 1
  
    Args:
        df: DataFrame con columna 'clase_ternaria'
        baja_2_1: Booleano que indica si se considera BAJA+1 como positivo
  
    Returns:
        pd.DataFrame: DataFrame con clase_ternaria convertida a valores binarios (0, 1)
    """

    #logger.info("Convirtiendo clase_ternaria a target binario")
  
    # Contar valores originales para logging (antes de modificar)

  
    # Modificar el DataFrame usando .loc para evitar SettingWithCopyWarning

  
    # Log de la conversión
  
    #logger.info(f"Conversión completada:")
    #logger.info(f"  Original - CONTINUA: {n_continua_orig}, BAJA+1: {n_baja1_orig}, BAJA+2: {n_baja2_orig}")
    #logger.info(f"  Binario - 0: {n_ceros}, 1: {n_unos}")
    #logger.info(f"  Distribución: {n_unos/(n_ceros + n_unos)*100:.2f}% casos positivos")

    return df

import pandas as pd
import os
import datetime
import logging


from src.config import *
from src.loader import cargar_datos, convertir_clase_ternaria_a_target
from src.features import feature_engineering_lag, feature_engineering_delta_lag
from src.optimization import optimizar

from src.best_params import cargar_mejores_hiperparametros

from src.test_evaluation import evaluar_en_test, guardar_resultados_test
from src.final_training import preparar_datos_entrenamiento_final, generar_predicciones_finales, entrenar_modelo_final
from src.output_manager import guardar_predicciones_finales


## config basico logging
os.makedirs(f"{BUCKET_NAME}/logs", exist_ok=True)

fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
nombre_log = f"log_{STUDY_NAME}_{fecha}.log"

log_path =os.path.join(f"{BUCKET_NAME}/logs/", nombre_log)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8-sig"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


## Funcion principal
def main():
    logger.info("Inicio de ejecucion.")

    #00 Cargar datos
    os.makedirs(f"{BUCKET_NAME}/data", exist_ok=True)
    data_path = os.path.join(BUCKET_NAME, DATA_PATH)
    print(data_path)
    df = cargar_datos(data_path)   

    #01 Feature Engineering
  
    #####
    #COMO SE QUE LES GUSTA MUCHO LO ANTERIOR Y LES LLEVA MUCHO TIEMPO 🙄
    #
    #Corroborando si existe antes
    #Cargar el df_fe
    #
    #Guardar el df_fe para no tener que hacerlo de nuevo
    #
    #####

    if os.path.exists(os.path.join(BUCKET_NAME, "data", f"df_fe{STUDY_NAME}.csv")):
        logger.info("✅ df_fe.csv encontrado")
        df_fe = pd.read_csv(os.path.join(BUCKET_NAME, "data", f"df_fe{STUDY_NAME}.csv"))
    else:
        logger.info("❌ df_fe.csv no encontrado")
        atributos = [col for col in df.columns if col.startswith(('c', 'm'))]
        atributos.remove("clase_ternaria")
        cant_lag = 2
        df_fe = feature_engineering_lag(df, atributos, cant_lag)
        df_fe = feature_engineering_delta_lag(df, atributos, cant_lag)
        logger.info(f"Feature Engineering completado: {df_fe.shape}")
        logger.info("Guardando df_fe.csv")
        df_fe.to_csv(os.path.join(BUCKET_NAME, "data", f"df_fe{STUDY_NAME}.csv"), index=False)


    #03 Ejecutar optimizacion de hiperparametros
    study = optimizar(df_fe, n_trials=100, undersampling=0.02)
  
    #04 Análisis adicional
    logger.info("=== ANÁLISIS DE RESULTADOS ===")
    trials_df = study.trials_dataframe()
    if len(trials_df) > 0:
        top_5 = trials_df.nlargest(5, 'value')
        logger.info("Top 5 mejores trials:")
        for idx, trial in top_5.iterrows():
            logger.info(f"  Trial {trial['number']}: {trial['value']:,.0f}")
  
    logger.info("=== OPTIMIZACIÓN COMPLETADA ===")

    logger.info(f">>> Ejecución finalizada. Revisar logs para mas detalles.")

if __name__ == "__main__":
    main()
