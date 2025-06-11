from flask import Flask, jsonify
from kmeans_analysis import perform_kmeans

app = Flask(__name__)

@app.route('/clusters')
def clusters():
    centroids, inertia, df, feature_names = perform_kmeans()
    result = {
        'centroids': centroids.to_dict(orient='records'),
        'inertia': inertia,
        'data': df.to_dict(orient='records'),
        'features': feature_names
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
