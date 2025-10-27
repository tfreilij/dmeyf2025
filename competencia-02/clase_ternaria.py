# -*- coding: utf-8 -*-
import duckdb
import pandas as pd

import yaml

with open("config-dev.yml", "r") as file:
    config = yaml.safe_load(file)

dataset_crudo_file = config["DATASET_CRUDO_PATH"]
dataset_ternaria = config["DATASET_TERNARIA_PATH"]

conn = duckdb.connect()

conn.execute(f"""
    CREATE OR REPLACE TABLE competencia_01_crudo AS
    SELECT *
    FROM read_csv_auto('{dataset_crudo_file}')
""")

# Check the loaded data
result = conn.execute("SELECT COUNT(*) FROM competencia_01_crudo").fetchone()
print(f"Loaded {result[0]} rows from {dataset_crudo_file}")

clase_ternaria_query = """
SELECT 
    t1.*,
    CASE
        WHEN t1.foto_mes IN ('202105', '202106') THEN NULL
        WHEN t1.foto_mes NOT IN ('202105', '202106')
             AND EXISTS (
                 SELECT 1
                 FROM competencia_01_crudo t2
                 WHERE t2.numero_de_cliente = t1.numero_de_cliente
                   AND CAST(t2.foto_mes AS INTEGER) = CAST(t1.foto_mes AS INTEGER) + 2
             )
        THEN 'CONTINUA'
        WHEN t1.foto_mes NOT IN ('202106')
             AND NOT EXISTS (
                 SELECT 1
                 FROM competencia_01_crudo t2
                 WHERE t2.numero_de_cliente = t1.numero_de_cliente
                   AND CAST(t2.foto_mes AS INTEGER) = CAST(t1.foto_mes AS INTEGER) + 1
             )
        THEN 'BAJA+1'
        ELSE 'BAJA+2'
    END AS clase_ternaria
FROM competencia_01_crudo t1
"""

result_df = conn.execute(clase_ternaria_query).df()
result_df.to_csv(dataset_ternaria, index=False)
print(f"Saved {len(result_df)} rows to {dataset_ternaria}")

summary = conn.execute("""
    SELECT foto_mes, clase_ternaria, COUNT(*) as count
    FROM (
        SELECT 
            t1.*,
            CASE
                WHEN t1.foto_mes IN ('202105', '202106') THEN NULL
                WHEN t1.foto_mes NOT IN ('202105', '202106')
                     AND EXISTS (
                         SELECT 1
                         FROM competencia_01_crudo t2
                         WHERE t2.numero_de_cliente = t1.numero_de_cliente
                           AND CAST(t2.foto_mes AS INTEGER) = CAST(t1.foto_mes AS INTEGER) + 2
                     )
                THEN 'CONTINUA'
                WHEN t1.foto_mes NOT IN ('202106')
                     AND NOT EXISTS (
                         SELECT 1
                         FROM competencia_01_crudo t2
                         WHERE t2.numero_de_cliente = t1.numero_de_cliente
                           AND CAST(t2.foto_mes AS INTEGER) = CAST(t1.foto_mes AS INTEGER) + 1
                     )
                THEN 'BAJA+1'
                ELSE 'BAJA+2'
            END AS clase_ternaria
        FROM competencia_01_crudo t1
    ) t
    GROUP BY foto_mes, clase_ternaria
    ORDER BY foto_mes ASC, clase_ternaria ASC
""").df()

print("\nClase ternaria distribution:")
print(summary)

# Close the connection
conn.close()

