#!/usr/bin/env python3
import polars as pl
import logging
import os
from load_config import Config
from pathlib import Path

logger = logging.getLogger(__name__)

config = Config()

def get_pesos_columns():
    df = pl.read_csv(config.__getitem__("DICCIONARIO_DATOS"))
    col_pesos = df.filter(pl.col("unidad") == "pesos").select(pl.col("campo"))
    logger.info(col_pesos["campo"].to_list())
    return col_pesos["campo"].to_list()

def generate_deltas(df : pl.DataFrame):

    query_deltas_pl = []
    for c in get_pesos_columns():
        if c in ["mtarjeta_visa_debitos_automaticos"]:
            continue
        delta_1 = pl.col(c) - pl.col(c).shift(1).over("numero_de_cliente")
        delta_2 = pl.col(c) - pl.col(c).shift(2).over("numero_de_cliente")
        sum_delta_2 = (pl.col(c) - pl.col(c).shift(1).over("numero_de_cliente")) + (pl.col(c).shift(1).over("numero_de_cliente") - pl.col(c).shift(2).over("numero_de_cliente"))

        query_deltas_pl.append(delta_1.alias(f"delta_1_{c}"))
        query_deltas_pl.append(delta_2.alias(f"delta_2_{c}"))
        query_deltas_pl.append(sum_delta_2.alias(f"sum_delta_{c}"))

    df = df.with_columns(query_deltas_pl)

    return df



def generate_feature_engineering_sql(diccionario_datos, table_name) -> str:
    """Generate the complete feature engineering SQL query dynamically"""
    logger.info("Generate Feature Engineering SQL")
    # List of columns to create delta features for
    columns = [
        'active_quarter', 'mrentabilidad', 'mrentabilidad_annual', 'mcomisiones', 'mactivos_margen',
        'mpasivos_margen', 'cproductos', 'ccuenta_corriente', 'mcuenta_corriente_adicional',
        'mcuenta_corriente', 'ccaja_ahorro', 'mcaja_ahorro', 'mcaja_ahorro_adicional',
        'mcaja_ahorro_dolares', 'cdescubierto_preacordado', 'mcuentas_saldo', 'ctarjeta_debito',
        'ctarjeta_debito_transacciones', 'mautoservicio', 'ctarjeta_visa', 'ctarjeta_visa_transacciones',
        'mtarjeta_visa_consumo', 'ctarjeta_master', 'ctarjeta_master_transacciones', 'mtarjeta_master_consumo',
        'cprestamos_personales', 'mprestamos_personales', 'cprestamos_prendarios', 'mprestamos_prendarios',
        'mprestamos_hipotecarios', 'cplazo_fijo', 'mplazo_fijo_dolares', 'mplazo_fijo_pesos',
        'cinversion1', 'minversion1_pesos', 'minversion1_dolares', 'cinversion2', 'minversion2',
        'ccaja_seguridad', 'cpayroll_trx', 'mpayroll', 'mpayroll2', 'cpayroll2_trx',
        'ccuenta_debitos_automaticos', 'mcuenta_debitos_automaticos', 'ctarjeta_visa_debitos_automaticos',
        'mttarjeta_visa_debitos_automaticos', 'ctarjeta_master_debitos_automaticos', 'mttarjeta_master_debitos_automaticos',
        'cpagodeservicios', 'mpagodeservicios', 'cpagomiscuentas', 'mpagomiscuentas',
        'ccajeros_propios_descuentos', 'mcajeros_propios_descuentos', 'ctarjeta_visa_descuentos',
        'mtarjeta_visa_descuentos', 'ctarjeta_master_descuentos', 'mtarjeta_master_descuentos',
        'ccomisiones_mantenimiento', 'mcomisiones_mantenimiento', 'ccomisiones_otras', 'mcomisiones_otras',
        'cforex', 'cforex_buy', 'mforex_buy', 'cforex_sell', 'mforex_sell',
        'ctransferencias_recibidas', 'mtransferencias_recibidas', 'ctransferencias_emitidas', 'mtransferencias_emitidas',
        'cextraccion_autoservicio', 'mextraccion_autoservicio', 'tcallcenter', 'ccallcenter_transacciones',
        'chomebanking_transacciones', 'ccajas_consultas', 'ccajas_extracciones', 'ccajas_otras',
        'catm_trx', 'matm', 'ctrx_quarter', 'cmobile_app_trx',
        'Master_delinquency', 'Master_mfinanciacion_limite', 'Master_Fvencimiento', 'Master_Finiciomora',
        'Master_msaldototal', 'Master_msaldopesos', 'Master_msaldodolares', 'Master_mconsumospesos',
        'Master_mconsumosdolares', 'Master_mlimitecompra', 'Master_madelantopesos', 'Master_madelantodolares',
        'Master_fultimo_cierre', 'Master_mpagado', 'Master_mpagospesos', 'Master_mpagosdolares',
        'Master_mconsumototal', 'Master_cconsumos', 'Master_cadelantosefectivo', 'Master_mpagominimo',
        'Visa_delinquency', 'Visa_mfinanciacion_limite', 'Visa_msaldototal', 'Visa_msaldopesos',
        'Visa_msaldodolares', 'Visa_mconsumospesos', 'Visa_mconsumosdolares', 'Visa_mlimitecompra',
        'Visa_madelantopesos', 'Visa_madelantodolares', 'Visa_fultimo_cierre', 'Visa_mpagado',
        'Visa_mpagospesos', 'Visa_mpagosdolares', 'Visa_mconsumototal', 'Visa_cconsumos'
    ]
        
    

def run_feature_engineering():
    """Run feature engineering with DuckDB using config file"""
      
    if not config:
        logger.error("Failed to load configuration")
        return
        
    df = pl.read_csv(os.path.join(config.__getitem__("DATASETS_PATH"),config.__getitem__("DATASET_TERNARIA_FILE")))

    df = generate_deltas(df)
    
    df.write_csv(os.path.join(config.__getitem__("BUCKET"),config.__getitem__("DATASET_FE_FILE")))

if __name__ == "__main__":
    run_feature_engineering()
