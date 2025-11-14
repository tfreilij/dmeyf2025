source ../../.venv312/bin/activate
mkdir ../../buckets/$2/log
python competencia-02.py > ../../buckets/$2/log/optimizacion-$1.txt 2>&1 &