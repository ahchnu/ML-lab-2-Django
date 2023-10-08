import os
import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


MODELS_PATH = os.path.join(os.getcwd(), 'mlweb', 'ai', 'models')
CSV_FILE_NAME = 'drug200.csv'
MODEL_MAX_COLUMN_COUNT = 5

AVAILABLE_MODELS = [
    {
        'name': 'Decision Tree Classifier',
        'file_name': 'decision_tree_classifier_model.pkl',
        'classifier': DecisionTreeClassifier()
    },
    {
        'name': 'K-Nearest Neighbors',
        'file_name': 'k_nearest_neighbors_model.pkl',
        'classifier': KNeighborsClassifier()
    },
    {
        'name': 'Support Vector Machine',
        'file_name': 'support_vector_machine_model.pkl',
        'classifier': SVC()
    }
]


with open(os.path.join(MODELS_PATH, 'le', 'Sex.pkl'), 'rb') as encoders_file:
    le_sex = pickle.load(encoders_file)

with open(os.path.join(MODELS_PATH, 'le', 'BP.pkl'), 'rb') as encoders_file:
    le_bp = pickle.load(encoders_file)

with open(os.path.join(MODELS_PATH, 'le', 'Cholesterol.pkl'), 'rb') as encoders_file:
    le_cholesterol = pickle.load(encoders_file)

with open(os.path.join(MODELS_PATH, 'le', 'Na_to_K.pkl'), 'rb') as encoders_file:
    le_Na_to_K = pickle.load(encoders_file)

with open(os.path.join(MODELS_PATH, 'le', 'Drug.pkl'), 'rb') as encoders_file:
    le_drug = pickle.load(encoders_file)


def get_model(df, model_name):
    _, file_name, classifier = get_model_by_name(model_name)
    # if we use all X, then we return prepared model
    if len(df.columns) == MODEL_MAX_COLUMN_COUNT:
        with open(os.path.join(MODELS_PATH, file_name), 'rb') as model_file:
            return pickle.load(model_file)
    # otherwise, we create new model
    df_drug = pd.read_csv(os.path.join(MODELS_PATH, CSV_FILE_NAME))

    df_drug['Sex'] = le_sex.transform(df_drug['Sex'])
    df_drug['BP'] = le_bp.transform(df_drug['BP'])
    df_drug['Cholesterol'] = le_cholesterol.transform(df_drug['Cholesterol'])
    df_drug['Drug'] = le_drug.transform(df_drug['Drug'])

    # drop excluded X
    columns_to_remove = set(df_drug.columns) - set(df.columns)
    columns_to_remove.remove('Drug')
    df_drug.drop(columns=columns_to_remove, inplace=True)

    # model training
    X = df_drug.drop("Drug", axis=1).values
    y = df_drug["Drug"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)

    classifier.fit(X_train, y_train)

    return classifier


def get_model_by_name(name):
    for model_data in AVAILABLE_MODELS:
        if model_data['name'] == name:
            return model_data['name'], model_data['file_name'], model_data['classifier']
    return None, None, None


def get_input_data(form_data):
    data = {}

    if form_data['include_age']:
        data['Age'] = [form_data['age']]
    if form_data['include_gender']:
        data['Sex'] = [form_data['gender']]
    if form_data['include_blood_pressure']:
        data['BP'] = [form_data['blood_pressure']]
    if form_data['include_cholesterol']:
        data['Cholesterol'] = [form_data['cholesterol']]
    if form_data['include_Na_to_K']:
        data['Na_to_K'] = [form_data['Na_to_K']]

    input_data = pd.DataFrame(data)

    if form_data['include_gender']:
        input_data['Sex'] = le_sex.transform(input_data['Sex'])
    if form_data['include_blood_pressure']:
        input_data['BP'] = le_bp.transform(input_data['BP'])
    if form_data['include_cholesterol']:
        input_data['Cholesterol'] = le_cholesterol.transform(input_data['Cholesterol'])

    return input_data


def inverse_y(predicted_label):
    return le_drug.inverse_transform([predicted_label])[0]


def get_data_count():
    df = pd.read_csv(os.path.join(MODELS_PATH, CSV_FILE_NAME))
    custom_csv_path = os.path.join(MODELS_PATH, 'custom_' + CSV_FILE_NAME)
    if not os.path.isfile(custom_csv_path):
        df_custom = pd.DataFrame(columns=df.columns)
        df_custom.to_csv(custom_csv_path, index=False)
    df_custom = pd.read_csv(custom_csv_path)
    return len(df) + len(df_custom)


def add_data(form_data):
    age = form_data['age']
    gender = form_data['gender']
    blood_pressure = form_data['blood_pressure']
    cholesterol = form_data['cholesterol']
    na_to_k = form_data['Na_to_K']
    drug = form_data['drug']

    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [gender],
        'BP': [blood_pressure],
        'Cholesterol': [cholesterol],
        'Na_to_K': [na_to_k],
        'Drug': [drug]
    })

    custom_csv_path = os.path.join(MODELS_PATH, 'custom_' + CSV_FILE_NAME)
    df = pd.read_csv(custom_csv_path)
    df = pd.concat([df, input_data], ignore_index=True)
    df.to_csv(custom_csv_path, index=False)
