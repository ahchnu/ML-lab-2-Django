from django import forms


class ModelValidationForm(forms.Form):
    age = forms.IntegerField()
    gender = forms.CharField()
    blood_pressure = forms.CharField()
    cholesterol = forms.CharField()
    Na_to_K = forms.FloatField()
