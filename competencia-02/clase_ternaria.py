# -*- coding: utf-8 -*-
import polars as pl
import os
import yaml
from load_config import Config


config = Config()
BUCKET = config["BUCKET"]
DATASET_CRUDO_FILE = config["DATASET_CRUDO_FILE"]
DATASET_TERNARIA_FILE = config["DATASET_TERNARIA_FILE"]
DATASETS_PATH = config["DATASETS_PATH"]

df_crudo = pl.read_csv(os.path.join(DATASETS_PATH,DATASET_CRUDO_FILE), infer_schema_length=None)

df_ternaria = df_crudo.with_columns(
    pl.col("foto_mes").shift(-1).over("numero_de_cliente").alias("next_month_foto_mes"),
    pl.col("foto_mes").shift(-2).over("numero_de_cliente").alias("next_two_months_foto_mes")
).with_columns(
    has_next_month = pl.col("next_month_foto_mes").is_not_null(),
    has_next_two_months = pl.col("next_two_months_foto_mes").is_not_null()
).with_columns(
    pl.when(pl.col("foto_mes") >= 202105)
    .then(pl.lit(None))
    .when(pl.col("has_next_two_months"))
    .then(pl.lit("CONTINUA"))
    .when(pl.col("has_next_month") & ~pl.col("has_next_two_months"))
    .then(pl.lit("BAJA+2"))
    .otherwise(pl.lit("BAJA+1"))
    .alias("clase_ternaria")
).drop(["next_month_foto_mes", "next_two_months_foto_mes", "has_next_month", "has_next_two_months"])


df_ternaria.write_csv(os.path.join(BUCKET,DATASET_TERNARIA_FILE))
