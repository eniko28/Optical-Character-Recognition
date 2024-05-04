import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def load_data(file_path):
    data = np.genfromtxt(file_path, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y

def euclidean_distance(x, y):
    euclidean = sqrt(np.sum(np.power(x - y, 2)))
    return euclidean

def cosine_similarity_custom(x, y):
    dot_product = np.dot(x, y)
    x_norm = np.sqrt(np.sum(np.power(x, 2)))
    y_norm = np.sqrt(np.sum(np.power(y, 2)))
    cosine_similarity = 1 - dot_product / (x_norm * y_norm)
    return cosine_similarity

class kNNClassifier:
    def __init__(self, k, distance_metric):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_data):
        predictions = []
        for data in X_data:
            distances = []
            for train_point in self.X_train:
                if self.distance_metric == 'euclidean':
                    distance = euclidean_distance(data, train_point)
                elif self.distance_metric == 'cosine':
                    distance = cosine_similarity_custom(data, train_point)
                distances.append(distance)
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_indices]
            prediction = np.bincount(nearest_labels).argmax()
            predictions.append(prediction)
        return predictions

class CentroidClassifier:
    def __init__(self, distance_metric):
        self.distance_metric = distance_metric

    def fit(self, X_train, y_train):
        self.centroids = {}
        unique_labels = np.unique(y_train)      # az osszes egyedi cimke
        for label in unique_labels:
            mask = y_train == label             # azonositom azokat az adatpontokat, melyek az adott osztalyhoz tartoznak
            X_label = X_train[mask]             # kivalasztom azokat az adatpontokat, amelyek az adott osztalyhoz tartoznak
            centroid = np.mean(X_label, axis=0) # kiszamolom a centrumot az oszlopok menten
            self.centroids[label] = centroid

    def predict(self, X_data):
        predictions = []
        for data in X_data:
            distances = {}
            if self.distance_metric == 'euclidean':
                for label, centroid in self.centroids.items():
                    distance = euclidean_distance(data, centroid)
                    distances[label] = distance
            elif self.distance_metric == 'cosine':
                for label, centroid in self.centroids.items():
                    distance = cosine_similarity_custom(data, centroid)
                    distances[label] = distance
            prediction = min(distances, key=distances.get)
            predictions.append(prediction)
        return predictions

    def plot_centroids(self):
        fig, axs = plt.subplots(2, 5, figsize=(12, 6))
        axs = axs.ravel()
        for i, (label, centroid) in enumerate(self.centroids.items()):
            axs[i].imshow(centroid.reshape(8, 8), cmap='gray')
            axs[i].set_title(f'Number: {label}')
            axs[i].axis('off')
        plt.tight_layout()
        plt.show()

    def plot_heatmap(self, X_test, y_test):
        distances_matrix = np.zeros((len(X_test), len(self.centroids)))
        sorted = np.argsort(y_test)
        X = X_test[sorted];

        for i, test_point in enumerate(X):
            for j, centroid in enumerate(self.centroids.values()):
                distances_matrix[i, j] = euclidean_distance(test_point, centroid)

        plt.figure(figsize=(10, 6))
        sns.heatmap(distances_matrix, cmap='coolwarm', xticklabels=self.centroids.keys(), yticklabels=self.centroids.keys())
        plt.xlabel('Centroids')
        plt.ylabel('Test Instances')
        plt.title('Euclidean Distances from Centroids')
        plt.show()

def accuracy(y_true, y_pred):
    classes = np.unique(y_true)
    accuracy_per_class = {}
    for cls in classes:
        correct_predictions = np.sum((y_true == cls) & (y_pred == cls))
        total_samples = np.sum(y_true == cls)
        accuracy_per_class[cls] = correct_predictions / total_samples if total_samples != 0 else 0
    return accuracy_per_class


def plot_cosine_similarity_heatmap(X_test, y_test):
    cosine_similarities = np.zeros((len(X_test), len(X_test)))

    for i, vector1 in enumerate(X_test):
        for j, vector2 in enumerate(X_test):
            cosine_similarities[i, j] = cosine_similarity_custom(vector1, vector2)

    cosine_simi = np.argsort(y_test)
    sorted_sim = cosine_similarities[cosine_simi,:][:,cosine_simi]
    plt.figure(figsize=(10, 6))
    sns.heatmap(sorted_sim, cmap='coolwarm', xticklabels=False, yticklabels=False)
    plt.title('Cosine Similarity Heatmap')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.show()

def main():

    X_train, y_train = load_data('optdigits.tra')
    X_test, y_test = load_data('optdigits.tes')

    knn_classifier_euclidean = kNNClassifier(k=5, distance_metric='euclidean')
    knn_classifier_euclidean.fit(X_train, y_train)
    knn_predictions_euclidean = knn_classifier_euclidean.predict(X_test)
    knn_accuracy_euclidean= accuracy(y_test, knn_predictions_euclidean)
    print('kNN: test pontossága (Euklideszi távolság):', knn_accuracy_euclidean)
    plot_confusion_matrix(y_test, knn_predictions_euclidean, 'kNN Confusion Matrix (Euclidean Distance)')

    knn_classifier_euclidean2 = kNNClassifier(k=5, distance_metric='euclidean')
    knn_classifier_euclidean2.fit(X_train, y_train)
    knn_predictions_euclidean2 = knn_classifier_euclidean.predict(X_train)
    knn_accuracy_euclidean2 = accuracy(y_train, knn_predictions_euclidean2)
    print('kNN: train pontossága (Euklideszi távolság):', knn_accuracy_euclidean2)

    knn_classifier_cosine = kNNClassifier(k=5, distance_metric='cosine')
    knn_classifier_cosine.fit(X_train, y_train)
    knn_predictions_cosine = knn_classifier_cosine.predict(X_test)
    knn_accuracy_cosine = accuracy(y_test, knn_predictions_cosine)
    print('kNN: test pontossága (Koszinusz hasonlóság):', knn_accuracy_cosine)
    plot_confusion_matrix(y_test, knn_predictions_cosine, 'kNN Confusion Matrix (Cosine Similarity)')

    knn_classifier_cosine2 = kNNClassifier(k=5, distance_metric='cosine')
    knn_classifier_cosine2.fit(X_train, y_train)
    knn_predictions_cosine2 = knn_classifier_cosine2.predict(X_train)
    knn_accuracy_cosine2 = accuracy(y_train, knn_predictions_cosine2)
    print('kNN: train pontossága (Koszinusz hasonlóság):', knn_accuracy_cosine2)

    centroid_classifier_euclidean = CentroidClassifier(distance_metric='euclidean')
    centroid_classifier_euclidean.fit(X_train, y_train)
    centroid_predictions_euclidean = centroid_classifier_euclidean.predict(X_test)
    centroid_accuracy_euclidean = accuracy(y_test, centroid_predictions_euclidean)
    print('Centroid: test pontossága (Euklideszi távolság):', centroid_accuracy_euclidean)
    plot_confusion_matrix(y_test, centroid_predictions_euclidean, 'Centroid Confusion Matrix (Euclidean Distance)')

    centroid_classifier_euclidean2 = CentroidClassifier(distance_metric='euclidean')
    centroid_classifier_euclidean2.fit(X_train, y_train)
    centroid_predictions_euclidean2 = centroid_classifier_euclidean2.predict(X_train)
    centroid_accuracy_euclidean2 = accuracy(y_train, centroid_predictions_euclidean2)
    print('Centroid: train pontossága (Euklideszi távolság):', centroid_accuracy_euclidean2)

    centroid_classifier_cosine = CentroidClassifier(distance_metric='cosine')
    centroid_classifier_cosine.fit(X_train, y_train)
    centroid_predictions_cosine = centroid_classifier_cosine.predict(X_test)
    centroid_accuracy_cosine = accuracy(y_test, centroid_predictions_cosine)
    print('Centroid: test pontossága (Koszinusz hasonlóság):', centroid_accuracy_cosine)
    plot_confusion_matrix(y_test, centroid_predictions_cosine, 'Centroid Confusion Matrix (Cosine Similarity)')

    centroid_classifier_cosine2 = CentroidClassifier(distance_metric='cosine')
    centroid_classifier_cosine2.fit(X_train, y_train)
    centroid_predictions_cosine2 = centroid_classifier_cosine2.predict(X_train)
    centroid_accuracy_cosine2 = accuracy(y_train, centroid_predictions_cosine2)
    print('Centroid: train pontossága (Koszinusz hasonlóság):', centroid_accuracy_cosine2)
    centroid_classifier = CentroidClassifier(distance_metric='euclidean')
    centroid_classifier.fit(X_train, y_train)
    centroid_classifier.plot_centroids()
    plot_cosine_similarity_heatmap(X_test, y_test)
    centroid_classifier.plot_heatmap(X_test, y_test)


if __name__ == '__main__':
    main()
