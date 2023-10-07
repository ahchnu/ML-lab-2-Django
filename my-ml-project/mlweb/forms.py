from django import forms


class ModelValidationForm(forms.Form):
    age = forms.IntegerField()
    gender = forms.CharField()
    blood_pressure = forms.CharField()
    cholesterol = forms.CharField()
    Na_to_K = forms.FloatField()


class CustomModelValidationForm(forms.Form):
    prediction_model = forms.CharField(required=False)
    age = forms.IntegerField(required=False)
    gender = forms.CharField(required=False)
    blood_pressure = forms.CharField(required=False)
    cholesterol = forms.CharField(required=False)
    Na_to_K = forms.FloatField(required=False)
    include_age = forms.BooleanField(required=False, initial=True)
    include_gender = forms.BooleanField(required=False, initial=True)
    include_blood_pressure = forms.BooleanField(required=False, initial=True)
    include_cholesterol = forms.BooleanField(required=False, initial=True)
    include_Na_to_K = forms.BooleanField(required=False, initial=True)
