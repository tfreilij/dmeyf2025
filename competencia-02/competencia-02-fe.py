#!/usr/bin/env python3
import duckdb
import pandas as pd
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            logger.info(f"✅ Loaded configuration from {config_path}")
            return config
    except Exception as e:
        logger.error(f"❌ Error loading config from {config_path}: {e}")
        return {}

def generate_feature_engineering_sql() -> str:
    """Generate the complete feature engineering SQL query dynamically"""
    
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
    
    for col in columns:
        # Delta 1 and 2 features
        delta_features.append(f"t1.{col} - lag(t1.{col}, 1) over (partition by t1.numero_de_cliente order by t1.foto_mes) as delta_1_{col}")
        delta_features.append(f"t1.{col} - lag(t1.{col}, 2) over (partition by t1.numero_de_cliente order by t1.foto_mes) as delta_2_{col}")
        
        # Sum of deltas features
        sum_delta_features.append(f"(t1.{col} - lag(t1.{col}, 1) over (partition by t1.numero_de_cliente order by t1.foto_mes)) + (t1.{col} - lag(t1.{col}, 2) over (partition by t1.numero_de_cliente order by t1.foto_mes)) as sum_deltas_{col}")
    
    # Build the complete SQL query
    sql_query = f"""
    CREATE OR REPLACE TABLE competencia_01_fe AS
    SELECT
      t1.*,
      {',\n      '.join(delta_features)},
      {',\n      '.join(sum_delta_features)}
    FROM competencia_01_fe t1
    ORDER BY t1.numero_de_cliente, t1.foto_mes
    """
    
    return sql_query

def run_feature_engineering():
    """Run feature engineering with DuckDB using config file"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load configuration
    config_path = "config-dev.yml"
    config = load_config(config_path)
    
    if not config:
        logger.error("Failed to load configuration")
        return
    
    # Get dataset path from config
    dataset_path = config.get("DATASET_TERNARIA_PATH")
    if not dataset_path:
        logger.error("No DATA_PATH found in config")
        return
    
    logger.info(f"Dataset path: {dataset_path}")
    
    # Initialize DuckDB connection
    conn = duckdb.connect()
    
    try:
        # Load the dataset
        logger.info("Loading dataset...")
        conn.execute(f"""
            CREATE OR REPLACE TABLE competencia_01_fe AS
            SELECT *
            FROM read_csv_auto('{dataset_path}')
        """)
        
        # Get basic info
        result = conn.execute("SELECT COUNT(*) as row_count FROM competencia_01_fe").fetchone()
        logger.info(f"✅ Loaded {result[0]} rows")
        
        # Execute feature engineering SQL
        logger.info("Executing feature engineering...")
        
        # Generate the complete feature engineering SQL query
        feature_sql = generate_feature_engineering_sql()
        
        conn.execute(feature_sql)
        
        # Export results
        output_path = config["DATASET_FE_PATH"]
        conn.execute(f"COPY competencia_01_fe TO '{output_path}' (FORMAT CSV, HEADER)")
        logger.info(f"✅ Exported results to {output_path}")
        
        # Show final info
        result = conn.execute("SELECT COUNT(*) as row_count FROM competencia_01_fe").fetchone()
        logger.info(f"✅ Final dataset: {result[0]} rows")
        
    except Exception as e:
        logger.error(f"❌ Error during feature engineering: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    run_feature_engineering()
