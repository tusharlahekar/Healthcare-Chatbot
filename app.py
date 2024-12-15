from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load data
train_df = pd.read_csv("D:\sample chatbot\Datasets\Training.csv")
test_df = pd.read_csv("D:\sample chatbot\Datasets\Testing.csv")

# Replace string labels with integers
prognosis_mapping = {  # mapping to integer
    'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3,
    'Drug Reaction': 4, 'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7,
    'Gastroenteritis': 8, 'Bronchial Asthma': 9, 'Hypertension ': 10,
    'Migraine': 11, 'Cervical spondylosis': 12, 'Paralysis (brain hemorrhage)': 13,
    'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16, 'Dengue': 17,
    'Typhoid': 18, 'hepatitis A': 19, 'Hepatitis B': 20, 'Hepatitis C': 21,
    'Hepatitis D': 22, 'Hepatitis E': 23, 'Alcoholic hepatitis': 24,
    'Tuberculosis': 25, 'Common Cold': 26, 'Pneumonia': 27,
    'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29,
    'Varicose veins': 30, 'Hypothyroidism': 31, 'Hyperthyroidism': 32,
    'Hypoglycemia': 33, 'Osteoarthristis': 34, 'Arthritis': 35,
    '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37,
    'Urinary tract infection': 38, 'Psoriasis': 39, 'Impetigo': 40
}

train_df.replace({'prognosis': prognosis_mapping}, inplace=True)
test_df.replace({'prognosis': prognosis_mapping}, inplace=True)

# Define features and labels
features = train_df.columns[:-1]  # all columns except prognosis
X_train = train_df[features]
y_train = train_df["prognosis"]
X_test = test_df[features]
y_test = test_df["prognosis"]

# Train models
def train_decision_tree():
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    return clf

def train_random_forest():
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    return clf

def train_naive_bayes():
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    return clf

# Predict disease function
def predict_disease(symptoms, model_type):
    symptom_vector = [0] * len(features)
    for symptom in symptoms:
        if symptom in features:
            index = features.get_loc(symptom)
            symptom_vector[index] = 1

    if model_type == 'decision_tree':
        clf = train_decision_tree()
    elif model_type == 'random_forest':
        clf = train_random_forest()
    elif model_type == 'naive_bayes':
        clf = train_naive_bayes()

    prediction = clf.predict([symptom_vector])
    for disease, index in prognosis_mapping.items():
        if index == prediction[0]:
            return disease
    return "Not Found"

@app.route('/')
def index():
    return render_template('index.html', symptoms=features.tolist())

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    name = data['name']
    age = data['age']
    symptoms = data['symptoms']
    model_type = data['model_type']
    prediction = predict_disease(symptoms, model_type)
    return jsonify({'name': name, 'age': age, 'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
