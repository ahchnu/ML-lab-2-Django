from sklearn.tree import DecisionTreeClassifier
from django.shortcuts import render
from .forms import CustomModelValidationForm, ModelValidationForm
from . import services


def index(request):
    return render(request, "mlweb/index.html")


def model(request):
    context = {
        'available_models': services.AVAILABLE_MODELS,
    }
    if request.method == 'GET':
        return render(request, "mlweb/ml.html", context)
    if request.method == 'POST':
        form = CustomModelValidationForm(request.POST)
        if form.is_valid():
            input_data = services.get_input_data(form.cleaned_data)

            prediction_model = services.get_model(input_data, form.cleaned_data['prediction_model'])

            prediction = prediction_model.predict(input_data)

            original_label = services.inverse_y(prediction[0])

            context['drug_type'] = original_label
            return render(request, "mlweb/ml_result.html", context)
        else:
            context['form'] = form
            return render(request, "mlweb/ml.html", context)


def add_data(request):
    context = {'data_count': services.get_data_count()}
    if request.method == 'GET':
        return render(request, 'mlweb/add_data.html', context)
    if request.method == 'POST':
        form = ModelValidationForm(request.POST)
        if form.is_valid():
            services.add_data(form.cleaned_data)
            context['data_count'] = services.get_data_count()
            return render(request, 'mlweb/add_data.html', context)
        else:
            context['form'] = form
            return render(request, "mlweb/add_data.html", context)
