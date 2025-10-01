# Malnutrition Reproducibility

Proyecto para reproducir experimentos de clasificación relacionados con desnutrición usando modelos CNN y un dataset sintético controlado. Incluye notebooks con pipelines completos, pesos preentrenados y un script para generar el dataset.

## Estructura
- Notebooks: `PRIMEROS_MODELOS_repro.ipynb`, `ULTIMOS_MODELOS_repro.ipynb`.
- Script de datos: `scripts/make_synthetic_dataset.py`.
- Datos: `synthetic_dataset/{train,val,test}` con archivos `0DS_*.png`, `1DS_*.png`, `2DS_*.png`, `3DS_*.png`.
- Modelos: `models/<modelo>/weights_*.{h5,tflite}` (p. ej., ResNet50, MobileNetV2, VGG16).
- Requisitos: `requirements.txt`.

## Requisitos e instalación
1) Crear entorno virtual (Windows PowerShell):
```
python -m venv .venv
.venv\Scripts\Activate.ps1
```
2) Instalar dependencias:
```
pip install -r requirements.txt
```

## Dataset
- Formato: imágenes RGB 255×255 sin subcarpetas por clase; la clase se infiere del prefijo del archivo (`0DS`, `1DS`, `2DS`, `3DS`).
- Opción 1: usar el ZIP incluido (`synthetic_dataset.zip`) y descomprimir en la raíz.
- Opción 2: generar desde el script:
```
python scripts/make_synthetic_dataset.py
```
Esto crea/valida `synthetic_dataset/{train,val,test}` y genera un `synthetic_dataset.zip` reproducible.

## Cómo ejecutar
- Abrir los notebooks en JupyterLab y seguir las celdas en orden:
```
jupyter lab
```
- `PRIMEROS_MODELOS_repro.ipynb`: primeros experimentos, baseline y preparación de datos.
- `ULTIMOS_MODELOS_repro.ipynb`: experimentos finales, comparación de arquitecturas y exportación de pesos.

## Reproducibilidad
- Se usan semillas fijas en data/entrenamiento. Mantenga el patrón de nombres `<CLASE>_<índice>.png` para consistencia.
- Pesos preentrenados disponibles en `models/` para evaluación rápida.

## Citar
Si usa este trabajo, cite el archivo `CITATION.cff` del repositorio.

## Licencia
Este proyecto se distribuye bajo la licencia del archivo `LICENSE`.
