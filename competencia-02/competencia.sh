echo "config=config-prod.yml" > .env
source ../../.venv312/bin/activate
python competencia-02.py > ../../buckets/$2/log/optimizacion-$1.txt 2>&1 &