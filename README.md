# Ejemplo de clustering con KMeans

Este repositorio incluye un script de ejemplo para realizar agrupamiento (clustering) usando `pandas` y `scikit-learn`.

## Requisitos

- Python 3
- pandas
- scikit-learn
- matplotlib
- flask

Puedes instalar las dependencias con:

```bash
pip install pandas scikit-learn matplotlib flask
```

## Uso

Ejecuta el script para cargar el conjunto de datos Iris, aplicar KMeans y mostrar un gráfico de los clusters:

```bash
python kmeans_analysis.py
```

Se imprimirán los centroides, la inercia del modelo y se mostrará una gráfica con los clusters obtenidos.

## Servidor web

También puedes iniciar un pequeño servidor Flask que expone los resultados en formato JSON y una página web para visualizarlos:

```bash
python app.py
```

Abre tu navegador en `http://localhost:5000` para ver la página `index.html` que consume los datos del endpoint `/clusters`.
