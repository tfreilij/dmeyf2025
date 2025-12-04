Para ejecutar primero se debe hacer el FE que incluye generación de clase ternaria y luego las nuevas variables.
Se puede poner en un .sh lo siguiente...

source /home/$1/.venv312/bin/activate

cd /home/$1/dmeyf2025/

git checkout competencia03

pip install -r requirements.txt

python competencia-03-fe.py


Luego el script hecho para notebook corre consumiendo competencia_03_fe.csv que estará en el bucket de origen.
Los resultados se guardarán en el bucket de destino (ver variable/diccionario CONFIG que está en las primeras
celdas del script.)