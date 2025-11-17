# -*- coding: utf-8 -*-
import polars as pl
import os
import logging
import datetime
from load_config import Config


config = Config()
BUCKETS = config["BUCKETS"]
BUCKET_TARGET = "b1"
DATASETS_PATH = "b1"

fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
nombre_log = f"log_clase_ternaria_{fecha}.log"
log_directory = os.path.join(BUCKETS,BUCKET_TARGET,"log",)
os.makedirs(log_directory, exist_ok=True)
log_path =os.path.join(log_directory, nombre_log)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8-sig"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

dataset_crudo_file = "competencia_02_crudo.csv.gz"
df_crudo = pl.read_csv(os.path.join(BUCKETS,BUCKET_TARGET,dataset_crudo_file), infer_schema_length=None)
df = df_crudo.sort(by=["numero_de_cliente", "foto_mes"], descending=[False, False])
df_ternaria = df.with_columns(
    pl.col("foto_mes").shift(-1).over("numero_de_cliente").alias("next_month_foto_mes"),
    pl.col("foto_mes").shift(-2).over("numero_de_cliente").alias("next_two_months_foto_mes")
).with_columns(
    has_next_month = pl.col("next_month_foto_mes").is_not_null(),
    has_next_two_months = pl.col("next_two_months_foto_mes").is_not_null()
).with_columns(
    pl.when(pl.col("foto_mes") >= 202107)
    .then(pl.lit(None))
    .when(pl.col("has_next_two_months"))
    .then(pl.lit("CONTINUA"))
    .when(pl.col("has_next_month") & ~pl.col("has_next_two_months"))
    .then(pl.lit("BAJA+2"))
    .otherwise(pl.lit("BAJA+1"))
    .alias("clase_ternaria")
).drop(["next_month_foto_mes", "next_two_months_foto_mes", "has_next_month", "has_next_two_months"])

logger.info(f"Resumen de clase ternaria")
resumen = df_ternaria['clase_ternaria'].value_counts()
logger.info(f"{resumen}")


resumen.write_csv(os.path.join(BUCKETS,BUCKET_TARGET,"resumen.csv"))
df_ternaria.write_csv(os.path.join(BUCKETS,BUCKET_TARGET,"competencia_02.csv"))
