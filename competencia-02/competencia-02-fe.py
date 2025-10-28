#!/usr/bin/env python3
import duckdb
import polars as pl
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

col_drops = {
          "numero_de_cliente", "foto_mes", "active_quarter", "clase_ternaria",
          "cliente_edad", "cliente_antiguedad",
          "Visa_fultimo_cierre", "Master_fultimo_cierre",
          "Visa_Fvencimiento", "Master_Fvencimiento"
      }


def load_config(config_path: str) -> dict:
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            logger.info(f"✅ Loaded configuration from {config_path}")
            return config
    except Exception as e:
        logger.error(f"❌ Error loading config from {config_path}: {e}")
        return {}

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
    
    # Generate delta features for each column
    delta_features = []
    sum_delta_features = []
    
    col_pesos = diccionario_datos.filter(pl.col("unidad") == "pesos").select(pl.col("campo"))
    query_deltas = ""
    for c in col_pesos["campo"].to_list():
        if c in ["mtarjeta_visa_debitos_automaticos"]:
            continue
        delta_1 = f"t1.{c} - lag(t1.{c}, 1) over (partition by t1.numero_de_cliente order by t1.foto_mes) as delta_1_{c} , \n"
        delta_2 = f"t1.{c} - lag(t1.{c}, 2) over (partition by t1.numero_de_cliente order by t1.foto_mes) as delta_2_{c} , \n"
        sum_delta_2 = f"(t1.{c} - lag(t1.{c}, 1) over (partition by t1.numero_de_cliente order by t1.foto_mes)) + (lag(t1.{c}, 1) over (partition by t1.numero_de_cliente order by t1.foto_mes) - lag(t1.{c}, 2) over (partition by t1.numero_de_cliente order by t1.foto_mes)) as sum_delta_{c},"

        query_deltas = query_deltas + delta_1 + delta_2 + sum_delta_2

    
    # Build the complete SQL query
    sql_query = f"""
    CREATE OR REPLACE TABLE {table_name} AS
    SELECT
        t1.*,
        {query_deltas}
    FROM {table_name} t1
    ORDER BY t1.numero_de_cliente, t1.foto_mes;
    """
    
    return sql_query

def run_feature_engineering():
    """Run feature engineering with DuckDB using config file"""
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    config_path = "config-dev.yml"
    config = load_config(config_path)
    
    if not config:
        logger.error("Failed to load configuration")
        return
    
    dataset_path = config.get("DATASET_TERNARIA_PATH")
    if not dataset_path:
        logger.error("No DATA_PATH found in config")
        return
    
    logger.info(f"Dataset path: {dataset_path}")
    
    conn = duckdb.connect()
    table_name = config.get("TABLE_NAME")
    try:
        logger.info("Loading dataset...")
        conn.execute(f"""
            CREATE OR REPLACE TABLE {table_name} AS
            SELECT *
            FROM read_csv_auto('{dataset_path}')
        """)

        result = conn.execute(f"SELECT COUNT(*) as row_count FROM {table_name}").fetchone()
        logger.info(f"✅ Loaded {result[0]} rows")
        
        logger.info("Executing feature engineering...")
        
        diccionario_datos = config.get("DICCIONARIO_DATOS")
        diccionario_datos = pl.read_csv(diccionario_datos)

        feature_sql = generate_feature_engineering_sql(diccionario_datos,table_name)
        
        logger.info("Execute Query " + feature_sql)
        conn.execute(feature_sql)
        
        output_path = config["DATASET_FE_PATH"]
        conn.execute(f"COPY {table_name} TO '{output_path}' (FORMAT CSV, HEADER)")
        logger.info(f"✅ Exported results to {output_path}")
        
        result = conn.execute(f"SELECT COUNT(*) as row_count FROM {table_name}").fetchone()
        logger.info(f"✅ Final dataset: {result[0]} rows")
        
    except Exception as e:
        logger.error(f"❌ Error during feature engineering: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    run_feature_engineering()
