import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def perform_kmeans():
    """Run KMeans on the iris dataset and return results."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    model = KMeans(n_clusters=3, random_state=42)
    model.fit(df)

    df["cluster"] = model.labels_

    centroids = pd.DataFrame(model.cluster_centers_, columns=iris.feature_names)
    inertia = model.inertia_

    return centroids, inertia, df, iris.feature_names


def main():
    centroids, inertia, df, feature_names = perform_kmeans()

    print("Centroides:")
    print(centroids)
    print("\nInercia:", inertia)

    plt.figure(figsize=(8, 6))
    plt.scatter(df[feature_names[0]], df[feature_names[1]], c=df["cluster"])
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title("Clusters con KMeans")
    plt.show()


if __name__ == '__main__':
    main()
