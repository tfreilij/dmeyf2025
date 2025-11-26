import polars as pl
import logging
from pathlib import Path
from load_config import Config
import datetime
import numpy as np

def setup_logging(log_path: Path):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8-sig"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)
def clase_ternaria(df: pl.DataFrame) -> pl.DataFrame:
    df = (
            df.with_columns(
        pl.col("foto_mes").shift(-1).over("numero_de_cliente").alias("next_month_foto_mes"),
        pl.col("foto_mes").shift(-2).over("numero_de_cliente").alias("next_two_months_foto_mes")
    ).with_columns(
        has_next_month = pl.col("next_month_foto_mes").is_not_null(),
        has_next_two_months = pl.col("next_two_months_foto_mes").is_not_null()
    ).with_columns(
        pl.when(pl.col("foto_mes") >= 202108)
        .then(pl.lit(None))
        .when(pl.col("has_next_two_months"))
        .then(pl.lit("CONTINUA"))
        .when(pl.col("has_next_month") & ~pl.col("has_next_two_months"))
        .then(pl.lit("BAJA+2"))
        .otherwise(pl.lit("BAJA+1"))
        .alias("clase_ternaria")
    ).drop(["next_month_foto_mes", "next_two_months_foto_mes", "has_next_month", "has_next_two_months"])
    )
    return df
def filter_foto_mes_range(df: pl.DataFrame, start_mes: int, end_mes: int) -> pl.DataFrame:
    return df.filter(~((pl.col("foto_mes") >= start_mes) & (pl.col("foto_mes") <= end_mes)))
def deltas_columns(df: pl.DataFrame):
    drop_cols = {
        "numero_de_cliente", "foto_mes", "clase_ternaria",
        "cliente_edad", "cliente_antiguedad",
        "Visa_fultimo_cierre", "Master_fultimo_cierre",
        "Visa_Fvencimiento", "Master_Fvencimiento",
        "Master_Finiciomora", "Visa_Finiciomora",
        "cliente_vip", "internet"
    }

    return [c for c in df.columns if c not in drop_cols]
def generate_deltas(df: pl.DataFrame, logger: logging.Logger) -> pl.DataFrame:
    df = df.sort(["numero_de_cliente", "foto_mes"])

    cols_lagueables = [
        c for c in df.columns
        if c not in ["numero_de_cliente", "foto_mes", "clase_ternaria"]
    ]

    for lag in [1, 2]:
        df = df.with_columns([
            pl.col(c).shift(lag).over("numero_de_cliente").alias(f"{c}_lag{lag}")
            for c in cols_lagueables
        ])

    df = df.with_columns([
        (pl.col(c) - pl.col(f"{c}_lag1")).alias(f"{c}_delta1")
        for c in cols_lagueables
    ] + [
        (pl.col(c) - pl.col(f"{c}_lag2")).alias(f"{c}_delta2")
        for c in cols_lagueables
    ])

    return df

def ctrx_quarter(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.when(pl.col("cliente_antiguedad") == 1)
        .then(pl.col("ctrx_quarter") * 5.0)
        .when(pl.col("cliente_antiguedad") == 2)
        .then(pl.col("ctrx_quarter") * 2.0)
        .when(pl.col("cliente_antiguedad") == 3)
        .then(pl.col("ctrx_quarter") * 1.2)
        .otherwise(pl.col("ctrx_quarter"))  # por si hay otros valores
        .alias("ctrx_quarter_normalizado")
    )
    return df

def drop_columns(df : pl.DataFrame) -> pl.DataFrame:
    
    cols = ['active_quarter',
 'cliente_vip',
 'tcuentas',
 'ccuenta_corriente',
 'mcuenta_corriente_adicional',
 'ctarjeta_master',
 'cprestamos_prendarios',
 'cprestamos_hipotecarios',
 'mplazo_fijo_pesos',
 'cinversion1',
 'minversion1_dolares',
 'cseguro_vida',
 'cseguro_auto',
 'cseguro_vivienda',
 'cseguro_accidentes_personales',
 'mpayroll2',
 'cpayroll2_trx',
 'ctarjeta_master_debitos_automaticos',
 'cpagodeservicios',
 'mpagodeservicios',
 'ccajeros_propios_descuentos',
 'mcajeros_propios_descuentos',
 'ctarjeta_visa_descuentos',
 'ctarjeta_master_descuentos',
 'mtarjeta_master_descuentos',
 'cforex',
 'cforex_buy',
 'mforex_buy',
 'cforex_sell',
 'mforex_sell',
 'ccheques_depositados',
 'mcheques_depositados',
 'ccheques_emitidos',
 'mcheques_emitidos',
 'ccheques_depositados_rechazados',
 'mcheques_depositados_rechazados',
 'ccheques_emitidos_rechazados',
 'mcheques_emitidos_rechazados',
 'ccajas_transacciones',
 'ccajas_depositos',
 'ccajas_extracciones',
 'ccajas_otras',
 'catm_trx_other',
 'Master_delinquency',
 'Master_mconsumosdolares',
 'Master_madelantopesos',
 'Master_madelantodolares',
 'Master_mpagosdolares',
 'Master_cadelantosefectivo',
 'Visa_delinquency',
 'Visa_madelantopesos',
 'Visa_madelantodolares',
 'Visa_cadelantosefectivo',
 'active_quarter_lag1',
 'cliente_vip_lag1',
 'tcuentas_lag1',
 'ccuenta_corriente_lag1',
 'mcuenta_corriente_adicional_lag1',
 'ctarjeta_visa_lag1',
 'ctarjeta_master_lag1',
 'cprestamos_prendarios_lag1',
 'mprestamos_prendarios_lag1',
 'cprestamos_hipotecarios_lag1',
 'mprestamos_hipotecarios_lag1',
 'cplazo_fijo_lag1',
 'mplazo_fijo_pesos_lag1',
 'cinversion1_lag1',
 'minversion1_dolares_lag1',
 'cinversion2_lag1',
 'cseguro_vida_lag1',
 'cseguro_auto_lag1',
 'cseguro_accidentes_personales_lag1',
 'mpayroll2_lag1',
 'cpayroll2_trx_lag1',
 'cpagodeservicios_lag1',
 'mpagodeservicios_lag1',
 'ccajeros_propios_descuentos_lag1',
 'mcajeros_propios_descuentos_lag1',
 'ctarjeta_visa_descuentos_lag1',
 'ctarjeta_master_descuentos_lag1',
 'mtarjeta_master_descuentos_lag1',
 'cforex_lag1',
 'cforex_buy_lag1',
 'mforex_buy_lag1',
 'cforex_sell_lag1',
 'mforex_sell_lag1',
 'ccheques_depositados_lag1',
 'mcheques_depositados_lag1',
 'ccheques_emitidos_lag1',
 'mcheques_emitidos_lag1',
 'ccheques_depositados_rechazados_lag1',
 'mcheques_depositados_rechazados_lag1',
 'ccheques_emitidos_rechazados_lag1',
 'mcheques_emitidos_rechazados_lag1',
 'ccajas_transacciones_lag1',
 'ccajas_depositos_lag1',
 'ccajas_otras_lag1',
 'Master_delinquency_lag1',
 'Master_status_lag1',
 'Master_Finiciomora_lag1',
 'Master_msaldodolares_lag1',
 'Master_mconsumosdolares_lag1',
 'Master_madelantopesos_lag1',
 'Master_madelantodolares_lag1',
 'Master_mpagosdolares_lag1',
 'Master_cadelantosefectivo_lag1',
 'Visa_delinquency_lag1',
 'Visa_madelantopesos_lag1',
 'Visa_madelantodolares_lag1',
 'Visa_cadelantosefectivo_lag1',
 'active_quarter_lag2',
 'cliente_vip_lag2',
 'ccuenta_corriente_lag2',
 'mcuenta_corriente_adicional_lag2',
 'ctarjeta_visa_lag2',
 'ctarjeta_master_lag2',
 'ctarjeta_master_transacciones_lag2',
 'cprestamos_prendarios_lag2',
 'mprestamos_prendarios_lag2',
 'mprestamos_hipotecarios_lag2',
 'cplazo_fijo_lag2',
 'mplazo_fijo_pesos_lag2',
 'cinversion1_lag2',
 'minversion1_pesos_lag2',
 'minversion1_dolares_lag2',
 'cseguro_vida_lag2',
 'cseguro_auto_lag2',
 'ccaja_seguridad_lag2',
 'mpayroll2_lag2',
 'cpayroll2_trx_lag2',
 'ctarjeta_master_debitos_automaticos_lag2',
 'cpagodeservicios_lag2',
 'mpagodeservicios_lag2',
 'ccajeros_propios_descuentos_lag2',
 'mcajeros_propios_descuentos_lag2',
 'ctarjeta_master_descuentos_lag2',
 'mtarjeta_master_descuentos_lag2',
 'cforex_lag2',
 'cforex_buy_lag2',
 'mforex_buy_lag2',
 'cforex_sell_lag2',
 'ccheques_depositados_lag2',
 'mcheques_depositados_lag2',
 'ccheques_emitidos_lag2',
 'mcheques_emitidos_lag2',
 'ccheques_depositados_rechazados_lag2',
 'mcheques_depositados_rechazados_lag2',
 'ccheques_emitidos_rechazados_lag2',
 'mcheques_emitidos_rechazados_lag2',
 'tcallcenter_lag2',
 'ccajas_transacciones_lag2',
 'ccajas_depositos_lag2',
 'ccajas_otras_lag2',
 'tmobile_app_lag2',
 'Master_delinquency_lag2',
 'Master_status_lag2',
 'Master_msaldodolares_lag2',
 'Master_mconsumosdolares_lag2',
 'Master_madelantopesos_lag2',
 'Master_madelantodolares_lag2',
 'Master_mpagosdolares_lag2',
 'Master_cadelantosefectivo_lag2',
 'Visa_delinquency_lag2',
 'Visa_status_lag2',
 'Visa_madelantopesos_lag2',
 'Visa_madelantodolares_lag2',
 'Visa_cadelantosefectivo_lag2',
 'active_quarter_delta1',
 'cliente_vip_delta1',
 'cliente_edad_delta1',
 'cliente_antiguedad_delta1',
 'tcuentas_delta1',
 'ccuenta_corriente_delta1',
 'mcuenta_corriente_adicional_delta1',
 'ctarjeta_visa_delta1',
 'ctarjeta_master_delta1',
 'cprestamos_prendarios_delta1',
 'cprestamos_hipotecarios_delta1',
 'mprestamos_hipotecarios_delta1',
 'cplazo_fijo_delta1',
 'mplazo_fijo_pesos_delta1',
 'cinversion1_delta1',
 'minversion1_pesos_delta1',
 'minversion1_dolares_delta1',
 'cinversion2_delta1',
 'minversion2_delta1',
 'cseguro_vida_delta1',
 'cseguro_auto_delta1',
 'cseguro_vivienda_delta1',
 'cseguro_accidentes_personales_delta1',
 'ccaja_seguridad_delta1',
 'cpayroll_trx_delta1',
 'mpayroll2_delta1',
 'cpayroll2_trx_delta1',
 'ctarjeta_master_debitos_automaticos_delta1',
 'cpagodeservicios_delta1',
 'mpagodeservicios_delta1',
 'ccajeros_propios_descuentos_delta1',
 'ctarjeta_master_descuentos_delta1',
 'mtarjeta_master_descuentos_delta1',
 'cforex_delta1',
 'cforex_buy_delta1',
 'mforex_buy_delta1',
 'cforex_sell_delta1',
 'ccheques_depositados_delta1',
 'mcheques_depositados_delta1',
 'ccheques_emitidos_delta1',
 'mcheques_emitidos_delta1',
 'ccheques_depositados_rechazados_delta1',
 'mcheques_depositados_rechazados_delta1',
 'ccheques_emitidos_rechazados_delta1',
 'mcheques_emitidos_rechazados_delta1',
 'tcallcenter_delta1',
 'ccajas_transacciones_delta1',
 'ccajas_depositos_delta1',
 'ccajas_extracciones_delta1',
 'cmobile_app_trx_delta1',
 'Master_delinquency_delta1',
 'Master_status_delta1',
 'Master_Fvencimiento_delta1',
 'Master_Finiciomora_delta1',
 'Master_msaldodolares_delta1',
 'Master_mconsumosdolares_delta1',
 'Master_madelantopesos_delta1',
 'Master_madelantodolares_delta1',
 'Master_fechaalta_delta1',
 'Master_cadelantosefectivo_delta1',
 'Visa_delinquency_delta1',
 'Visa_Finiciomora_delta1',
 'Visa_madelantopesos_delta1',
 'Visa_madelantodolares_delta1',
 'Visa_fechaalta_delta1',
 'Visa_cadelantosefectivo_delta1',
 'active_quarter_delta2',
 'cliente_vip_delta2',
 'cliente_antiguedad_delta2',
 'tcuentas_delta2',
 'ccuenta_corriente_delta2',
 'mcuenta_corriente_adicional_delta2',
 'mcaja_ahorro_adicional_delta2',
 'ctarjeta_visa_delta2',
 'ctarjeta_master_delta2',
 'cprestamos_prendarios_delta2',
 'cprestamos_hipotecarios_delta2',
 'mprestamos_hipotecarios_delta2',
 'mplazo_fijo_pesos_delta2',
 'cinversion1_delta2',
 'minversion1_pesos_delta2',
 'minversion1_dolares_delta2',
 'cinversion2_delta2',
 'minversion2_delta2',
 'cseguro_auto_delta2',
 'cseguro_vivienda_delta2',
 'cseguro_accidentes_personales_delta2',
 'mpayroll2_delta2',
 'cpayroll2_trx_delta2',
 'ctarjeta_master_debitos_automaticos_delta2',
 'cpagodeservicios_delta2',
 'mpagodeservicios_delta2',
 'mcajeros_propios_descuentos_delta2',
 'ctarjeta_visa_descuentos_delta2',
 'ctarjeta_master_descuentos_delta2',
 'mtarjeta_master_descuentos_delta2',
 'cforex_delta2',
 'cforex_buy_delta2',
 'mforex_buy_delta2',
 'cforex_sell_delta2',
 'mforex_sell_delta2',
 'ccheques_depositados_delta2',
 'mcheques_depositados_delta2',
 'ccheques_emitidos_delta2',
 'mcheques_emitidos_delta2',
 'ccheques_depositados_rechazados_delta2',
 'mcheques_depositados_rechazados_delta2',
 'ccheques_emitidos_rechazados_delta2',
 'mcheques_emitidos_rechazados_delta2',
 'ccajas_depositos_delta2',
 'cmobile_app_trx_delta2',
 'Master_delinquency_delta2',
 'Master_status_delta2',
 'Master_Finiciomora_delta2',
 'Master_msaldodolares_delta2',
 'Master_mconsumosdolares_delta2',
 'Master_madelantopesos_delta2',
 'Master_madelantodolares_delta2',
 'Master_mpagosdolares_delta2',
 'Master_cadelantosefectivo_delta2',
 'Visa_delinquency_delta2',
 'Visa_status_delta2',
 'Visa_Finiciomora_delta2',
 'Visa_madelantopesos_delta2',
 'Visa_madelantodolares_delta2',
 'Visa_cadelantosefectivo_delta2']

    drops = []
    for c in cols:
        if c in df.columns:
            drops.append(c)
            
    df = df.drop(drops)
    return df

def canaritos(df : pl.DataFrame) -> pl.DataFrame:
    
    n = df.height 
    return df.with_columns([
        pl.Series(f"canarito_{i}", np.random.rand(n))
        for i in range(1, 11)
    ])

def run_feature_engineering():
    config = Config()

    BUCKETS = config["BUCKETS"]
    BUCKET_ORIGIN = config["BUCKET_ORIGIN"]
    BUCKET_TARGET = config["BUCKET_ORIGIN"]

    file_origin_comp2 = Path(BUCKETS) / BUCKET_ORIGIN / "competencia_02_crudo.csv.gz"
    file_origin_comp3 = Path(BUCKETS) / BUCKET_ORIGIN / "competencia_03_crudo.csv.gz"
    file_target = Path(BUCKETS) / BUCKET_TARGET / "competencia_03_fe.csv"

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    log_dir = Path(BUCKETS) / BUCKET_TARGET / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"log_fe_{timestamp}.log"

    logger = setup_logging(log_path)

    logger.info(f"Reading dataset {file_origin_comp2}")
    df_comp2 = pl.read_csv(file_origin_comp2, infer_schema_length=None)

    logger.info(f"Reading dataset {file_origin_comp3}")
    df_comp3 = pl.read_csv(file_origin_comp3, infer_schema_length=None)

    df = df_comp2.vstack(df_comp3)
    
    logger.info("Sorting data")
    df = df.sort(["numero_de_cliente", "foto_mes"])

    logger.info("Generating clase_ternaria")
    df = clase_ternaria(df)

    logger.info("Drop Columns")
    df = drop_columns(df)
    logger.info("Generating deltas")
    df = generate_deltas(df, logger)

    logger.info("Delete 202006 month")
    df = filter_foto_mes_range(df,202006,202006)


    logger.info("CTRX_Quarter adjustment:")
    df = ctrx_quarter(df)

    logger.info("Canaritos:")
    df = canaritos(df)

    logger.info("Clase ternaria distribution:")
    logger.info(df["clase_ternaria"].value_counts())

    logger.info("Writing output...")
    df.write_csv(file_target)

    logger.info(f"Dataset written to {file_target}")


if __name__ == "__main__":
    run_feature_engineering()
