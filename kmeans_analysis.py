import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def main():
    # Cargar datos de ejemplo
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)

    # Crear modelo KMeans con 3 clusters
    model = KMeans(n_clusters=3, random_state=42)
    model.fit(X)

    # Añadir las etiquetas de cluster a los datos
    X['cluster'] = model.labels_

    # Mostrar centroides e inercia
    print('Centroides:')
    print(pd.DataFrame(model.cluster_centers_, columns=iris.feature_names))
    print('\nInercia:', model.inertia_)

    # Graficar dos de las variables para ver los clusters
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[iris.feature_names[0]], X[iris.feature_names[1]], c=X['cluster'])
    plt.xlabel(iris.feature_names[0])
    plt.ylabel(iris.feature_names[1])
    plt.title('Clusters con KMeans')
    plt.show()


if __name__ == '__main__':
    main()
