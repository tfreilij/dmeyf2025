import polars as pl
import logging
from pathlib import Path
from load_config import Config
import datetime

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
        df
        .with_columns([
            pl.col("foto_mes").shift(-1).over("numero_de_cliente").alias("next_month_foto_mes"),
            pl.col("foto_mes").shift(-2).over("numero_de_cliente").alias("next_two_months_foto_mes"),
        ])
        .with_columns([
            pl.col("next_month_foto_mes").is_not_null().alias("has_next_month"),
            pl.col("next_two_months_foto_mes").is_not_null().alias("has_next_two_months"),
        ])
        .with_columns(
            pl.when(pl.col("foto_mes") >= 202108)
            .then(pl.lit(None))
            .when(pl.col("has_next_two_months"))
            .then("CONTINUA")
            .when(pl.col("has_next_month") & ~pl.col("has_next_two_months"))
            .then("BAJA+2")
            .otherwise("BAJA+1")
            .alias("clase_ternaria")
        )
        .drop([
            "next_month_foto_mes", "next_two_months_foto_mes",
            "has_next_month", "has_next_two_months"
        ])
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
    cols = deltas_columns(df)
    numeric_cols = [c for c in cols if df.schema[c].is_numeric()]

    exprs = []
    for c in numeric_cols:
        logger.info(f"Generating deltas for column {c}")

        exprs.extend([
            pl.col(c).shift(1).over("numero_de_cliente").alias(f"lag_1_{c}").cast(pl.Float64),
            pl.col(c).shift(2).over("numero_de_cliente").alias(f"lag_2_{c}").cast(pl.Float64),
            (pl.col(c) - pl.col(c).shift(1).over("numero_de_cliente")).alias(f"delta_1_{c}").cast(pl.Float64),
            (pl.col(c) - pl.col(c).shift(2).over("numero_de_cliente")).alias(f"delta_2_{c}").cast(pl.Float64),
        ])

    return df.with_columns(exprs)

def run_feature_engineering():
    config = Config()

    BUCKETS = config["BUCKETS"]
    BUCKET_ORIGIN = config["BUCKET_ORIGIN"]
    BUCKET_TARGET = config["BUCKET_ORIGIN"]

    file_origin = Path(BUCKETS) / BUCKET_ORIGIN / "competencia_03_crudo.csv.gz"
    file_target = Path(BUCKETS) / BUCKET_TARGET / "competencia_03_fe.parquet"

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    log_dir = Path(BUCKETS) / BUCKET_TARGET / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"log_fe_{timestamp}.log"

    logger = setup_logging(log_path)

    logger.info(f"Reading dataset {file_origin}")
    df = pl.read_csv(file_origin, infer_schema_length=None)

    logger.info("Sorting data")
    df = df.sort(["numero_de_cliente", "foto_mes"])

    logger.info("Generating clase_ternaria")
    df = clase_ternaria(df)

    logger.info("Generating deltas")
    df = generate_deltas(df, logger)

    logger.info("Delete 202006 month")
    df = filter_foto_mes_range(df,202006,202006)

    logger.info("Clase ternaria distribution:")
    logger.info(df["clase_ternaria"].value_counts())

    logger.info("Writing parquet output...")
    df.write_parquet(file_target)

    logger.info(f"Dataset written to {file_target}")


if __name__ == "__main__":
    run_feature_engineering()
