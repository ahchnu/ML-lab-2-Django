import os
import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


MODELS_PATH = os.path.join(os.getcwd(), 'mlweb', 'ai', 'models')
MODEL_MAX_COLUMN_COUNT = 5


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


def get_model(df):
    # if we use all X, then we return prepared model
    if len(df.columns) == MODEL_MAX_COLUMN_COUNT:
        with open(os.path.join(MODELS_PATH, 'decision_tree_classifier_model.pkl'), 'rb') as model_file:
            return pickle.load(model_file)
    # otherwise, we create new model
    df_drug = pd.read_csv(os.path.join(MODELS_PATH, 'drug200.csv'))

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

    prediction_model = DecisionTreeClassifier()
    prediction_model.fit(X_train, y_train)

    return prediction_model


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
