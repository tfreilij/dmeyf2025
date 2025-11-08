echo "config=config-prod.yml" > .env
source ../../.venv312/bin/activate
python competencia-02.py > ../../buckets/logs/optimizacion-$1.txt 2>&1 &