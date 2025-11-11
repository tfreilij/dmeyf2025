#!/usr/bin/env python3
import polars as pl
import logging
import os
from load_config import Config
from pathlib import Path
import datetime

logger = logging.getLogger(__name__)

config = Config()
BUCKETS = config["BUCKETS"]
BUCKET_ORIGIN = config["BUCKET_ORIGIN"]
BUCKET_TARGET = config["BUCKET_TARGET"]

file_origin = os.path.join(BUCKETS,BUCKET_ORIGIN, "competencia_02.csv")
file_target = os.path.join(BUCKETS,BUCKET_TARGET, "competencia_02_fe.csv")

fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
nombre_log = f"log_fe_{fecha}.log"
log_path =os.path.join(BUCKETS,BUCKET_TARGET,"log", nombre_log)
os.makedirs(log_path, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(log_path, mode="w", encoding="utf-8-sig"),
        logging.StreamHandler()
    ]
)

def filter_foto_mes_range(df : pl.DataFrame, start_mes : int, end_mes : int):
    df = df.filter(~( (pl.col("foto_mes") >= start_mes) & (pl.col("foto_mes") <= end_mes)))
    return df


def deltas_columns(df : pl.DataFrame):


    col_drops = [
          "clase_ternaria","numero_de_cliente", "foto_mes", "active_quarter",
          "cliente_antiguedad", 
          "Visa_Fvencimiento", "Master_Fvencimiento", "Master_Finiciomora",
          "cliente_vip", "internet", "cliente_edad", "mrentabilidad_annual"
    ]

    columnas_despreciables = ['ccuenta_corriente', "tcuentas", "mpasivos_margen", 
    "mcuenta_corriente_adicional", "ccaja_ahorro", "mcaja_ahorro_adicional",
    "cprestamos_personales","mprestamos_personales", "mprestamos_prendarios","cprestamos_hipotecarios","mprestamos_hipotecarios"
    ,"cinversion1","minversion1_dolares","cinversion2","minversion2","cseguro_auto","cseguro_vivienda","cseguro_accidentes_personales",
    "ccaja_seguridad","mpayroll2","cpayroll2_trx","ccuenta_debitos_automaticos",
    "ctarjeta_visa_debitos_automaticos","ctarjeta_master_debitos_automaticos",
    "mttarjeta_master_debitos_automaticos","cpagodeservicios","mpagodeservicios","ccajeros_propios_descuentos","mcajeros_propios_descuentos"
    ,"ctarjeta_visa_descuentos","mtarjeta_visa_descuentos","cforex_buy","mforex_buy","ccheques_depositados","ccheques_emitidos",
    "ccheques_depositados_rechazados","mcheques_depositados_rechazados","ccheques_emitidos_rechazados","mcheques_emitidos_rechazados","ccajas_transacciones",
                        "ccajas_consultas","ccajas_depositos","ccajas_otras","Visa_fultimo_cierre","Master_fultimo_cierre","minversion1_pesos"]

    col_drops.extend(columnas_despreciables)

    all_cols = df.columns
    for c in col_drops:
        all_cols.remove(c)
    return all_cols

def generate_deltas(df : pl.DataFrame):

    expressions = []
    all_cols = deltas_columns(df)

    numeric_cols = [col for col in all_cols if df[col].dtype != pl.String]
    for c in numeric_cols:
        logger.info(f"Delta for column {c}")
        expressions.append(
            pl.col(c).shift(-1).over("numero_de_cliente").alias(f"lag_1_{c}").cast(pl.Float64)
        )
        expressions.append(
            pl.col(c).shift(-2).over("numero_de_cliente").alias(f"lag_2_{c}").cast(pl.Float64)
        )

        expressions.append(
            (pl.col(c) - pl.col(c).shift(-1).over("numero_de_cliente")).alias(f"delta_1_{c}").cast(pl.Float64)
        )
        expressions.append(
            (pl.col(c) - pl.col(c).shift(-2).over("numero_de_cliente")).alias(f"delta_2_{c}").cast(pl.Float64)
        )

    return df.with_columns(expressions)
       
def run_feature_engineering():   
    if not config:
        logger.error("Failed to load configuration")
        return
    
    logger.info(f"Reading dataset {file_origin}")
    df = pl.read_csv(file_origin)
    df = df.sort(by=["numero_de_cliente", "foto_mes"], descending=[False, True])
    logger.info("Generating deltas")
    df = generate_deltas(df)
    logger.info("Writing dataset")
    df.write_csv(file_target)
    logger.info(f"Dataset written to {file_target}")

if __name__ == "__main__":
    run_feature_engineering()
