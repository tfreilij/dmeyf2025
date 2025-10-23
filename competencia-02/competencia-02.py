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
```

## 2) Convirtiendo en binario el target segun entrenameinto, validacion o test

```python
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
```

## 3)Estructura del mian.py

```python
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
  
    #05 Test en mes desconocido
    #logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")


    #mejores_params = cargar_mejores_hiperparametros()
  
    # Evaluar en test
    #ganancia_test = evaluar_en_test(df_fe, mejores_params, undersampling=0.02)
  
    # Guardar resultados de test
    #guardar_resultados_test(ganancia_test[0])
  
    # Resumen de evaluación en test
    #logger.info("=== RESUMEN DE EVALUACIÓN EN TEST ===")
    #logger.info(f"✅ Ganancia en test: {ganancia_test[0]:,.0f}")
  
    #logger.info("=== GRAFICO DE TEST ===")
    #ruta_grafico = generar_grafico_test_completo(df_fe, tiradas=30, undersampling=0.02)
    #logger.info(f"✅ Gráfico generado: {ruta_grafico}")
  
    #06 Entrenar modelo final
    #logger.info("=== ENTRENAMIENTO FINAL ===")
 
    # Entrenar modelo final
    #logger.info("Entrenar modelo final")
    #modelo_final = entrenar_modelo_final(df_fe, mejores_params, undersampling=0.02)
  
    # Generar predicciones finales
    #logger.info("Generar predicciones finales")
    #df_predict = df_fe[df_fe['foto_mes'].isin([FINAL_PREDIC])]
    #clientes_predict = df_predict['numero_de_cliente'].values
    #X_predict = df_predict.drop(columns=['clase_ternaria'])
    #resultados = generar_predicciones_finales(modelo_final, X_predict, clientes_predict)

    # Resumen final
    #logger.info("=== RESUMEN FINAL ===")
    #logger.info(f"✅ Entrenamiento final completado exitosamente")
    #logger.info(f"📊 Mejores hiperparámetros utilizados: {mejores_params}")
    #logger.info(f"🎯 Períodos de entrenamiento: {FINAL_TRAIN}")
    #logger.info(f"🔮 Período de predicción: {FINAL_PREDIC}")
    #logger.info(f"📁 Archivo de salida: {archivo_salida}")
    #logger.info(f"📝 Log detallado: logs/{nombre_log}")
    #logger.info(f"📝 Resultados guardados en: {json_path}")
    logger.info(f">>> Ejecución finalizada. Revisar logs para mas detalles.")

if __name__ == "__main__":
    main()
```

---

## 2) Configuración de instancia en GCP

### **Crear VM en Google Cloud Platform**

**Desde la consola de GCP:**

1. Ir a **Compute Engine** > **VM instances**
2. Click **"Create Instance"**
3. Configurar:

```
Nombre: ml-optuna-vm
Región: us-central1
Tipo de máquina: e2-standard-4 (4 vCPUs, 16 GB RAM)
Sistema operativo: Ubuntu 22.04 LTS
Disco: 50 GB SSD
```

4. Click **"Create"**

---

## 3) Conexión SSH a GCP

### **Opción A: SSH desde consola web (Mas Facil pero no acepta los atajos de teclado)**

En GCP Console: **Compute Engine** > **VM instances** > Click botón **"SSH"**

### **Opción B: SSH desde terminal (Recomendado)**

**Usando gcloud (más simple):**

```bash
# Conectar a la instancia
gcloud compute ssh ml-optuna-vm --zone=us-central1-a

# Primera conexión generará claves SSH automáticamente
```

---

## 4) Gestión de sesiones con TMUX

### **¿Por qué TMUX?**

- Mantiene procesos corriendo tras desconexión SSH
- Múltiples ventanas y paneles
- Ideal para entrenamientos largos en GCP

### **Instalación y comandos básicos**

```bash
#Ya esta instalado

# GESTIÓN DE SESIONES
tmux new -s optimizacion          # Crear sesión
tmux ls                            # Listar sesiones
tmux attach -t optimizacion        # Conectar a sesión
# Ctrl+B, luego D                  # Desconectar (sesión sigue corriendo)
tmux kill-session -t optimizacion  # Eliminar sesión

# DIVIDIR PANELES
# Ctrl+B, luego "     # Dividir horizontal
# Ctrl+B, luego %     # Dividir vertical
# Ctrl+B, luego ←↑→↓  # Navegar entre paneles
# Ctrl+D o exit       # Cerrar panel

# VENTANAS MÚLTIPLES
# Ctrl+B, luego C     # Nueva ventana
# Ctrl+B, luego 0-9   # Cambiar a ventana N
# Ctrl+B, luego N     # Siguiente ventana
# Ctrl+B, luego P     # Ventana anterior
# Ctrl+B, luego ,     # Renombrar ventana
```

### **Layout recomendado**

```bash
# Crear sesión
tmux new -s ml_proyecto

# Layout en 3 paneles:
# ┌────────────────┬────────────┐
# │  Código        │   htop     │
# │  Terminal      │            │
# ├────────────────┴────────────┤
# │         Logs                │
# └─────────────────────────────┘
```

---

## 5) Monitoreo de recursos

### **HTOP - Monitor interactivo**

```bash
# Instalar
sudo apt install htop -y

# Ejecutar Ya esta instalado
htop

```

### **Comandos de monitoreo**

```bash
# CPU
mpstat -P ALL 1        # Uso por núcleo

# Memoria
free -h                # Resumen memoria
cat /proc/meminfo      # Detalle completo

# Procesos Python
ps aux | grep python
top -p $(pgrep -d',' python)

# Información del sistema
lscpu                  # Info CPUs
uname -a               # Sistema operativo
```


## 6) Ejecución del proyecto

### **Flujo de trabajo completo con TMUX**

```bash
# 1. Conectar por SSH

# 2. Crear/conectar sesión TMUX
tmux new -s optuna_run
# O si existe: tmux attach -t optuna_run

# 3. Activar entorno virtual
cd Project_Wendsday
source .venv/bin/activate

# 4. Dividir paneles
# Ctrl+B %  (vertical split)
# En panel derecho: htop

# Panel inferior: ejecutar
python main.py

# 5. Desconectar sin matar proceso
# Ctrl+B D

# 6. Cerrar SSH (proceso sigue corriendo en TMUX)
exit

# 7. Reconectar más tarde
ssh gcp-ml
tmux attach -t optuna_run

# 8. Ver resultados
ls -lh resultados/
ls -lh bd_Optuna/
```

### **Ejecutar proceso largo de forma segura**

```bash
# Opción 1: Con TMUX (recomendado)
tmux new -s optimization
source .venv/bin/activate
python main.py
# Ctrl+B D para desconectar

# Opción 2: Con nohup (si no tienes TMUX)
nohup python main.py > salida.log 2>&1 &

# Ver proceso
ps aux | grep python

# Ver logs en tiempo real
tail -f salida.log

# Matar proceso si es necesario
pkill -f main.py
```

---

## 7) Google Cloud Storage (Buckets)

Los **Buckets** son almacenamiento de objetos en GCP, ideales para guardar resultados, modelos y datasets.

### **Crear Bucket**

**Desde consola web:**

1. Ir a **Cloud Storage** > **Buckets**
2. Click **"Create".  YA ESTA TODO CREADO**
3. Configurar:

```
Nombre: ml-optuna-results-{tu-nombre}
Ubicación: us-central1 (misma región que VM)
Storage class: Standard
Access control: Uniform
```

### **Subir archivos al Bucket**

```bash
# Subir un archivo
gsutil cp resultados/mejores_params.json gs://ml-optuna-results-joaquin/

# Subir carpeta completa
gsutil -m cp -r resultados/ gs://ml-optuna-results-joaquin/resultados/

# Subir base de datos de Optuna
gsutil -m cp -r bd_Optuna/ gs://ml-optuna-results-joaquin/bd_Optuna/

# Subir logs
gsutil -m cp logs/*.log gs://ml-optuna-results-joaquin/logs/
```

### **Descargar archivos del Bucket**

```bash
# Descargar un archivo
gsutil cp gs://ml-optuna-results-joaquin/resultados/mejores_params.json ./

# Descargar carpeta completa
gsutil -m cp -r gs://ml-optuna-results-joaquin/resultados/ ./resultados/

# Sincronizar (solo archivos nuevos/modificados)
gsutil -m rsync -r ./resultados/ gs://ml-optuna-results-joaquin/resultados/
```

### **COMO TODO EN INFORMATICA ESTA LA FORMA VISUAL POR LA WEB EN ESTE CASO, APRA INTERACTUAR CON GCP, PERO SI QUEREMOS PODEMOS ISNTALAR gsutil Y TRABAJAR DESDE CONSOLA.**

---

## 📦 Dependencias necesarias

```bash
# requirements.txt (actualizar)

```

```bash
# Instalar en entorno virtual
source .venv/bin/activate
pip install -r requirements.txt
```


