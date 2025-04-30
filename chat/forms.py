from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django import forms
from captcha.fields import CaptchaField

# chat/forms.py
from django import forms
from captcha.fields import CaptchaField

class LoginForm(forms.Form):
    username = forms.CharField(
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter Username',
            'autocomplete': 'off'
        })
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter Password',
            'autocomplete': 'off'
        })
    )
    captcha = CaptchaField(
        label='CAPTCHA Verification',
        error_messages={'invalid': 'Invalid CAPTCHA. Please try again.'}
    )

class SignupForm(UserCreationForm):
    email = forms.EmailField(required=True, widget=forms.EmailInput(attrs={'class': 'form-control'}))
    username = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control'}))
    password1 = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control'}))
    password2 = forms.CharField(widget=forms.PasswordInput(attrs={'class': 'form-control'}))


class Meta:
        model = User
        fields = ['username','email', 'password1', 'password2']

from django import forms
from .models import PrivateMessage

class ChatMessageForm(forms.ModelForm):
    class Meta:
        model = PrivateMessage
        fields = ['content', 'file']  # Include file field

