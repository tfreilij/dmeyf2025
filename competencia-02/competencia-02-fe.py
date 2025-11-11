#!/usr/bin/env python3
import polars as pl
import logging
import os
from load_config import Config
from pathlib import Path
import datetime



logger = logging.getLogger(__name__)


config = Config()

BUCKET = config["BUCKET"]

os.makedirs(f"{BUCKET}/logs", exist_ok=True)

fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
nombre_log = f"log_fe_{fecha}.log"

log_path =os.path.join(f"{BUCKET}/logs/", nombre_log)



def filter_foto_mes_range(df : pl.DataFrame, start_mes : int, end_mes : int):
    df = df.filter(~( (pl.col("foto_mes") >= start_mes) & (pl.col("foto_mes") <= end_mes)))
    return df


def deltas_columns(df : pl.DataFrame):


    col_drops = [
          "numero_de_cliente", "foto_mes", "active_quarter",
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

    for c in all_cols:
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
    """Run feature engineering with DuckDB using config file"""
      
    if not config:
        logger.error("Failed to load configuration")
        return
    
    logger.info(f"Reading dataset from {os.path.join(config.__getitem__("BUCKET"),config.__getitem__("DATASET_TERNARIA_FILE"))}")
    df = pl.read_csv(os.path.join(config.__getitem__("BUCKET"),config.__getitem__("DATASET_TERNARIA_FILE")))
    df = df.sort(by=["numero_de_cliente", "foto_mes"], descending=[False, True])
    logger.info(f"After filtering: {df.shape} rows")
    logger.info("Generating deltas")
    df = generate_deltas(df)
    logger.info("Writing dataset")
    df.write_csv(os.path.join(config.__getitem__("BUCKET"),config.__getitem__("DATASET_FE_FILE")))
    logger.info(f"Dataset written to {os.path.join(config.__getitem__("BUCKET"),config.__getitem__("DATASET_FE_FILE"))}")

if __name__ == "__main__":
    run_feature_engineering()
