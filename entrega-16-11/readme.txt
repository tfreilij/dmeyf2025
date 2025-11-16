Se detallan variables para la notebook.

MES_TRAIN : Lista de foto_mes para optimización.

ejemplo [202002,202003, 202004, 202005, 202006, 202007, 202008, 202009, 202010,202011,202012,202101]

IS_EXPERIMENTO : Booleano. 
Indica si es experimento, se usa para saber que al predecir se pueden estimar ganancias idealizadas.

FINAL_PREDICT : Foto_mes final que se quiere predecir. 

FINAL_TRAIN : Lista de foto_mes para hacer la predicción final.

SEMILLA : Lista de semillas

UNDERSAMPLE_FRACTION : Float para indicar porcentaje de undersampling del dataset de train.

RUN_BAYESIAN_OPTIMIZATION : Booleano para indicar si se quiere correr una nueva optimización. En caso de ser falso se usará la que esté persistida.

N_TRIALS : Cantidad de trials para la optimización

APLICAR_UNDERSAMPLING : Booleano que indica si se quiere o no aplicar el undersampling