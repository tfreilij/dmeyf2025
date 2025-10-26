#!/usr/bin/env python3
import duckdb
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class DuckDBFeatureEngineering:
    
    def __init__(self, db_path: str = None):
        """
        Initialize DuckDB connection
        
        Args:
            db_path: Path to DuckDB database file (None for in-memory)
        """
        self.conn = duckdb.connect(db_path) if db_path else duckdb.connect()
        self.db_path = db_path
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def close(self):
        if self.conn:
            self.conn.close()
    
    def load_csv(self, csv_path: str, table_name: str = "competencia_02_fe") -> None:
        """
        Load CSV file into DuckDB table using read_csv_auto
        
        Args:
            csv_path: Path to CSV file
            table_name: Name of the table to create
        """
        try:
            # Use DuckDB's read_csv_auto for automatic type inference
            self.conn.execute(f"""
                CREATE OR REPLACE TABLE {table_name} AS
                SELECT *
                FROM read_csv_auto('{csv_path}')
            """)
            
            # Get basic info about loaded data
            result = self.conn.execute(f"SELECT COUNT(*) as row_count FROM {table_name}").fetchone()
            logger.info(f"✅ Loaded {result[0]} rows into table '{table_name}' from {csv_path}")
            
        except Exception as e:
            logger.error(f"❌ Error loading CSV from {csv_path}: {e}")
            raise
    
    def execute_sql(self, full_query: str, table_name: str = "competencia_02_fe") -> None:
        """
        Execute SQL query and replace table
        
        Args:
            sql_query: SQL query to execute
            table_name: Name of the table to create/replace
        """
        try:          
            self.conn.execute(full_query)
            
            result = self.conn.execute(f"SELECT COUNT(*) as row_count FROM {table_name}").fetchone()
            logger.info(f"✅ SQL executed successfully. Result: {result[0]} rows in '{table_name}'")
            
        except Exception as e:
            logger.error(f"❌ Error executing SQL: {e}")
            logger.error(f"Query: {full_query}")
            raise
    
    def export_to_csv(self, table_name: str, output_path: str) -> None:
        """
        Export table to CSV file
        
        Args:
            table_name: Name of table to export
            output_path: Output CSV file path
        """
        try:
            self.conn.execute(f"COPY {table_name} TO '{output_path}' (FORMAT CSV, HEADER)")
            logger.info(f"✅ Exported table '{table_name}' to {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Error exporting to CSV: {e}")
            raise
    
    def get_table_info(self, table_name: str) -> dict:
        """
        Get basic information about a table
        
        Args:
            table_name: Name of the table
            
        Returns:
            dict: Table information
        """
        try:
            # Get row count
            row_count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            
            # Get column info
            columns = self.conn.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
            """).fetchall()
            
            return {
                'row_count': row_count,
                'columns': columns,
                'column_count': len(columns)
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting table info: {e}")
            return {}
    
    def preview_table(self, table_name: str, limit: int = 5) -> pd.DataFrame:
        """
        Preview table data
        
        Args:
            table_name: Name of the table
            limit: Number of rows to preview
            
        Returns:
            pd.DataFrame: Preview of the table
        """
        try:
            return self.conn.execute(f"SELECT * FROM {table_name} LIMIT {limit}").df()
        except Exception as e:
            logger.error(f"❌ Error previewing table: {e}")
            return pd.DataFrame()

def simple_feature_engineering_example():
    """Example of how to use the DuckDBFeatureEngineering class"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Example usage
    with DuckDBFeatureEngineering() as fe:
        
        # Example: Load a CSV file (replace with your actual path)
        # fe.load_csv("path/to/your/data.csv", "competencia_01_fe")
        
        # For demo, create sample data
        logger.info("Creating sample data for demonstration...")
        import numpy as np
        
        sample_data = pd.DataFrame({
            'numero_de_cliente': range(1, 1001),
            'foto_mes': np.random.choice([202101, 202102, 202103, 202104], 1000),
            'cpayroll_trx': np.random.poisson(3, 1000),
            'ctrx_quarter': np.random.poisson(8, 1000),
            'clase_ternaria': np.random.choice(['CONTINUA', 'BAJA+1', 'BAJA+2'], 1000, p=[0.7, 0.2, 0.1])
        })
        
        # Register sample data as table
        fe.conn.register('competencia_01_fe', sample_data)
        
        # Example SQL feature engineering
        feature_engineering_sql = """
        SELECT 
            *,
            -- Lag features
            LAG(cpayroll_trx, 1) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) as cpayroll_trx_lag1,
            LAG(cpayroll_trx, 2) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) as cpayroll_trx_lag2,
            
            -- Delta features
            cpayroll_trx - LAG(cpayroll_trx, 1) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) as cpayroll_trx_delta1,
            
            -- Rolling statistics
            AVG(cpayroll_trx) OVER (
                PARTITION BY numero_de_cliente 
                ORDER BY foto_mes 
                ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
            ) as cpayroll_trx_avg3m,
            
            -- Ratio features
            CASE 
                WHEN ctrx_quarter > 0 THEN cpayroll_trx / ctrx_quarter 
                ELSE 0 
            END as payroll_to_trx_ratio,
            
            -- Time features
            EXTRACT(MONTH FROM foto_mes) as mes,
            
            -- Binary features
            CASE WHEN cpayroll_trx > 0 THEN 1 ELSE 0 END as has_payroll
        FROM competencia_01_fe
        """
        
        # Execute feature engineering
        fe.execute_sql(feature_engineering_sql, "competencia_01_fe")
        
        # Get table info
        info = fe.get_table_info("competencia_01_fe")
        logger.info(f"Table info: {info['row_count']} rows, {info['column_count']} columns")
        
        # Preview the data
        preview = fe.preview_table("competencia_01_fe", limit=3)
        logger.info("Preview of engineered data:")
        logger.info(f"\n{preview}")
        
        # Export to CSV
        fe.export_to_csv("competencia_01_fe", "engineered_features.csv")
        
        logger.info("✅ Feature engineering example completed!")

if __name__ == "__main__":
    simple_feature_engineering_example()
