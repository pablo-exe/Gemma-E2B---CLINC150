# Recorrido educativo de las fases 1 y 2

Estos cuadernos muestran el mismo experimento que la aplicación, celda a celda.
Cada uno es autocontenido: no importa `gemma_clinc` ni lee los archivos YAML.
La descarga de datos, los prompts, el modelo, el parser, las métricas y el bucle
experimental se escriben directamente en las celdas. Solo se importan
bibliotecas estándar y externas. Los parámetros iniciales replican los YAML
actuales; si cambian en el futuro, hay que actualizar el cuaderno correspondiente.

1. Desde la raíz del repositorio: `uv sync --extra gpu --extra train` y
   `uv pip install --python .venv/Scripts/python.exe ipykernel`.
2. Abra el repositorio en Jupyter o VS Code y seleccione el intérprete de `.venv`.
3. Ejecute `phase1_zero_shot.ipynb` o `phase2_qlora_sft.ipynb` en orden.

Se necesita acceso al modelo restringido de Hugging Face (`uv run hf auth login`)
y una GPU NVIDIA compatible. El primer uso descarga los pesos y el conjunto de
datos. Las rutas se resuelven desde la raíz del repositorio aunque Jupyter abra
el cuaderno dentro de `notebooks/`.

Los límites de muestra aparecen al principio de cada cuaderno. Su valor inicial
es pequeño para facilitar una primera ejecución; `None` ejecuta el experimento
completo. Con los mismos parámetros y límites, el cuaderno usa las mismas
entradas, orden de muestras, semillas, operaciones y formato de salida que el
comando correspondiente. Los resultados numéricos de GPU pueden variar
ligeramente entre ejecuciones. Los adaptadores y predicciones se guardan bajo
`artifacts/phase1/` o `artifacts/phase2/`, y se pueden leer con las herramientas
de la aplicación. Cada ejecución crea un directorio nuevo.

La fase 1 **no entrena**: mide el modelo original sobre `test`. La fase 2 entrena
con `train`, selecciona el adaptador con `val` y no consulta `test`. Después de
congelar la configuración y elegir el mejor adaptador, se puede evaluar una vez
en `test` con `gemma-clinc evaluate-adapter`.
