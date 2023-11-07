import tkinter as tk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# Load the dataset
data = pd.read_csv('sample.csv')

# Preprocess the text data using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
features = vectorizer.fit_transform(data['SentimentText'])

# Define the models to use
knn = KNeighborsClassifier()
nb = MultinomialNB()
svm = LinearSVC()

# Perform 10-fold cross-validation to determine the best model
knn_scores = cross_val_score(knn, features, data['Sentiment'], cv=10)
nb_scores = cross_val_score(nb, features, data['Sentiment'], cv=10)
svm_scores = cross_val_score(svm, features, data['Sentiment'], cv=10)

# Calculate the mean accuracy of each model
knn_accuracy = knn_scores.mean()
nb_accuracy = nb_scores.mean()
svm_accuracy = svm_scores.mean()

# Select the best model based on cross-validation
best_model = svm if svm_accuracy > max(knn_accuracy, nb_accuracy) else nb if nb_accuracy > max(knn_accuracy, svm_accuracy) else knn

# Train the selected model on the entire dataset
best_model.fit(features, data['Sentiment'])

# Define the UI elements
root = tk.Tk()
root.title("Sentiment Analysis")
root.geometry("400x300")

label = tk.Label(root, text="Enter text:")
label.pack()

entry = tk.Entry(root, width=50)
entry.pack()

output_label = tk.Label(root, text="")
output_label.pack()

# Define the function to perform sentiment analysis
def analyze_sentiment():
    new_text = entry.get()
    new_features = vectorizer.transform([new_text])
    predicted_sentiment = best_model.predict(new_features)[0]
    output_label.config(text="Predicted sentiment: " + str(predicted_sentiment))

# Add a button to trigger the sentiment analysis
button = tk.Button(root, text="Analyze", command=analyze_sentiment)
button.pack()

# Start the UI loop
root.mainloop()
