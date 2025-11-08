echo "config=config-dev.yml" > .env
source ../../.venv312/bin/activate
touch ../../logs/optimizacion-$1.txt
python competencia-02.py > ../../logs/optimizacion-$1.txt 2>&1 &