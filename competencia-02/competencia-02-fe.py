#!/usr/bin/env python3
import polars as pl
import logging
import os
from load_config import Config
from pathlib import Path

logger = logging.getLogger(__name__)

config = Config()

def filter_foto_mes_range(df : pl.DataFrame, start_mes : int, end_mes : int):
    df = df.filter(~( (pl.col("foto_mes") >= start_mes) & (pl.col("foto_mes") <= end_mes)))
    return df


def deltas_columns(df : pl.DataFrame):


    col_drops = {
          "numero_de_cliente", "foto_mes", "active_quarter", "clase_ternaria",
          "cliente_edad", "cliente_antiguedad",
          "Visa_fultimo_cierre", "Master_fultimo_cierre",
          "Visa_Fvencimiento", "Master_Fvencimiento", "Master_Finiciomora",
          "cliente_vip", "internet", "cliente_edad", "cliente_antiguedad", "mrentabilidad_annual", "clase_ternaria", 
      }
    all_cols = df.columns
    for c in col_drops:
        all_cols.remove(c)
    return all_cols



def generate_deltas(df : pl.DataFrame):

    all_cols = deltas_columns(df)

    for c in all_cols:
        df = df.with_columns(pl.col(c).alias(f"delta_1_{c}").cast(pl.Float64))
        df = df.with_columns(pl.col(c).alias(f"delta_2_{c}").cast(pl.Float64))
        df = df.with_columns(pl.col(c).alias(f"sum_delta_{c}").cast(pl.Float64))


    delta_cols = [f"delta_1_{c}" for c in all_cols ] + \
             [f"delta_2_{c}" for c in all_cols ] + \
             [f"sum_delta_{c}" for c in all_cols ]
    query_deltas_pl = []
    #for c in col_pesos["campo"].to_list():
    for c in delta_cols:
        print(f"Deltas for column {c}")
        delta_1 = (pl.col(c) - pl.col(c).shift(-1).over("numero_de_cliente")).cast(pl.Float64).alias(f"delta_1_{c}")
        delta_2 = (pl.col(c) - pl.col(c).shift(-2).over("numero_de_cliente")).cast(pl.Float64).alias(f"delta_2_{c}")
        sum_delta_2 = ((pl.col(c) - pl.col(c).shift(-1).over("numero_de_cliente")) + (pl.col(c).shift(-1).over("numero_de_cliente") - pl.col(c).shift(-2).over("numero_de_cliente"))).cast(pl.Float64).alias(f"sum_delta_{c}")

        query_deltas_pl.append(delta_1)
        query_deltas_pl.append(delta_2)
        query_deltas_pl.append(sum_delta_2)

    df = df.with_columns(query_deltas_pl)

    return df
       
def run_feature_engineering():
    """Run feature engineering with DuckDB using config file"""
      
    if not config:
        logger.error("Failed to load configuration")
        return
    
    logger.info(f"Reading dataset from {os.path.join(config.__getitem__("BUCKET"),config.__getitem__("DATASET_TERNARIA_FILE"))}")
    df = pl.read_csv(os.path.join(config.__getitem__("BUCKET"),config.__getitem__("DATASET_TERNARIA_FILE")))
    logger.info("Filtering foto_mes range")
    logger.info(f"Filtering from {202003} to {202012} : {df.shape} rows")
    df = filter_foto_mes_range(df, 202003, 202012)
    logger.info(f"After filtering: {df.shape} rows")
    logger.info("Generating deltas")
    df = generate_deltas(df)
    logger.info("Writing dataset")
    df.write_csv(os.path.join(config.__getitem__("BUCKET"),config.__getitem__("DATASET_FE_FILE")))
    logger.info(f"Dataset written to {os.path.join(config.__getitem__("BUCKET"),config.__getitem__("DATASET_FE_FILE"))}")

if __name__ == "__main__":
    run_feature_engineering()
