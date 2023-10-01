import os
import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from django.shortcuts import render
from .forms import ModelValidationForm

models_path = os.path.join(os.getcwd(), 'mlweb', 'ai', 'models')


def index(request):
    return render(request, "mlweb/index.html")


# with open(os.path.join(models_path, 'decision_tree_classifier_model.pkl'), 'rb') as model_file:
#     model = pickle.load(model_file)

with open(os.path.join(models_path, 'le', 'Sex.pkl'), 'rb') as encoders_file:
    le_sex = pickle.load(encoders_file)

with open(os.path.join(models_path, 'le', 'BP.pkl'), 'rb') as encoders_file:
    le_bp = pickle.load(encoders_file)

with open(os.path.join(models_path, 'le', 'Cholesterol.pkl'), 'rb') as encoders_file:
    le_cholesterol = pickle.load(encoders_file)

with open(os.path.join(models_path, 'le', 'Na_to_K.pkl'), 'rb') as encoders_file:
    le_Na_to_K = pickle.load(encoders_file)

with open(os.path.join(models_path, 'le', 'Drug.pkl'), 'rb') as encoders_file:
    le_drug = pickle.load(encoders_file)


def model(request):
    if request.method == 'GET':
        return render(request, "mlweb/ml.html")
    if request.method == 'POST':
        form = ModelValidationForm(request.POST)
        if form.is_valid():
            age = form.cleaned_data['age']
            gender = form.cleaned_data['gender']
            blood_pressure = form.cleaned_data['blood_pressure']
            cholesterol = form.cleaned_data['cholesterol']
            na_to_k = form.cleaned_data['Na_to_K']

            input_data = pd.DataFrame({
                'Age': [age],
                'Sex': [gender],
                'BP': [blood_pressure],
                'Cholesterol': [cholesterol],
                'Na_to_K': [na_to_k]
            })

            input_data['Sex'] = le_sex.transform(input_data['Sex'])
            input_data['BP'] = le_bp.transform(input_data['BP'])
            input_data['Cholesterol'] = le_cholesterol.transform(input_data['Cholesterol'])
            # input_data['Na_to_K'] = le_Na_to_K.transform(input_data['Na_to_K'])

            with open(os.path.join(models_path, 'decision_tree_classifier_model.pkl'), 'rb') as model_file:
                model = pickle.load(model_file)

            prediction = model.predict(input_data)

            predicted_label = prediction[0]
            original_label = le_drug.inverse_transform([predicted_label])[0]

            context = {"drug_type": original_label}
            return render(request, "mlweb/ml_result.html", context)
        else:
            context = {"error_message": "Invalid form"}
            return render(request, "mlweb/ml.html", context)
