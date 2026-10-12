# Guía del proyecto — Johann Vera

El objetivo es anticipar la elevación del agua en Crabtree Creek entre 1 y 6 horas, conservando los picos de creciente. El archivo no incluye caudal; la elevación está en pies sobre NAVD88, no en profundidad del río.

## Qué se corrigió

Se separaron períodos cronológicamente, se conservaron picos sin suavizar, se evitaron entradas futuras y se expresaron errores en unidades físicas. Los faltantes de lluvia no se transforman en ceros. Las horas EST/EDT se convierten a UTC antes de unir archivos.

La primera versión entrenada usa árboles de gradient boosting como comparación reproducible. Se incluye una versión corregida de LSTM, GRU y BiLSTM para ejecutar con TensorFlow, pero esas redes todavía no fueron entrenadas aquí. No se ha demostrado una mejora respecto a tu modelo original; sí se comparó con persistencia.

## Qué muestran los resultados

A 1 hora el error medio baja aproximadamente 29% frente a persistencia y el error en niveles altos baja aproximadamente 41%. A 4–6 horas el error promedio empeora, aunque el error de niveles altos mejora. Las seis horas siguen siendo un objetivo de investigación.

El umbral 210,87 ft NAVD88 es estadístico, calculado con entrenamiento. No es un umbral oficial de inundación. Los indicadores de detección son por observación de cinco minutos; no equivalen a detectar tormentas independientes ni prueban seis horas útiles de aviso.

Las gráficas alinean pronósticos sucesivos por su hora válida. Para una alerta real se necesita evaluar lo que se sabía en cada hora de emisión y cuándo apareció la primera advertencia antes del cruce observado.

## Cómo ejecutar

1. Abrir `Crabtree_Forecasting.ipynb` o seguir los comandos del README.
2. Descargar datos con `download_data.py` o usar los cinco archivos RDB incluidos en la copia completa.
3. Entrenar con `forecast.py` y evaluar con `evaluate_events.py`.
4. Consultar `results/metrics.csv`, `event_metrics.csv`, `threshold_metrics.csv` y la gráfica.

El siguiente aporte más útil será lluvia y niveles aguas arriba, pronósticos de lluvia archivados a la hora de emisión, y los umbrales operativos en el mismo datum. Antes de usar acumulaciones de lluvia se debe confirmar que cada lectura representa el total del intervalo de cinco minutos.
