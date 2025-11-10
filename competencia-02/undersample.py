from token import SEMI
import polars as pl
import numpy as np
import logging
import os
import datetime

from load_config import Config

logger = logging.getLogger(__name__)

config = Config()
BUCKET = config["BUCKET"]
RUN_BAYESIAN_OPTIMIZATION = config["RUN_BAYESIAN_OPTIMIZATION"]
UNDERSAMPLING_FRACTION = config["UNDERSAMPLING_FRACTION"]
DATASET_FE_FILE = config["DATASET_FE_FILE"]
DATASET_UNDERSAMPLED_FILE =  config["DATASET_UNDERSAMPLED_FILE"]
MES_TRAIN = config["MES_TRAIN"]

train_test_models = config["TRAIN_TEST_MODELS"]

os.makedirs(f"{BUCKET}/log", exist_ok=True)

fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
nombre_log = f"log_undersample_{fecha}.log"
log_path =os.path.join(f"{BUCKET}/log/", nombre_log)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8-sig"),
        logging.StreamHandler()
    ]
)

def generate_clase_binaria(df : pl.DataFrame):

    df = df.with_columns(pl.lit(0).alias('clase_binaria'))

    df = df.with_columns(
        pl.when(pl.col('clase_ternaria').is_in(['BAJA+2','BAJA+1'])).then(pl.lit(1)).otherwise(pl.lit(0)).alias('clase_binaria')
    )

    return df


logger.info(f"Config : {config}")

logger.info("Read DataFrame")
df = pl.read_csv(os.path.join(BUCKET,DATASET_FE_FILE))

df = df.filter(pl.col("foto_mes").is_in(MES_TRAIN))

logger.info("Generate Clase Binaria")
df = generate_clase_binaria(df)

logger.info(f"Undersampling Continuas with fraction : {UNDERSAMPLING_FRACTION} , DF shape : {df.shape}")

clientes_solo_continuas = df.group_by("numero_de_cliente").agg(n_bajas=pl.col("clase_binaria").sum()).filter(pl.col("n_bajas") == 0)
clientes_continua = clientes_solo_continuas['numero_de_cliente']
clientes_solo_continuas_undersampled = clientes_solo_continuas.sample(fraction=1-UNDERSAMPLING_FRACTION, seed=1000)
df = df.filter(~pl.col('numero_de_cliente').is_in(clientes_solo_continuas_undersampled["numero_de_cliente"]))
logger.info(f"DF shape after undersampling: {df.shape}")

logger.info("Writing dataset")
df.write_csv(os.path.join(config.__getitem__("BUCKET"),DATASET_UNDERSAMPLED_FILE))
logger.info(f"Dataset written to {os.path.join(config.__getitem__("BUCKET"),config.__getitem__("DATASET_FE_FILE"))}")