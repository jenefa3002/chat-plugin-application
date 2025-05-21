import os
from datetime import timedelta

from django.utils import timezone
from django.utils.timezone import now

from . import models
from .forms import LoginForm
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '2'
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Q
from django.dispatch import receiver
from django.shortcuts import redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout, user_logged_in, user_logged_out
from django.contrib.auth.forms import UserCreationForm
from django.views import View
from keras.src.utils import pad_sequences
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from .models import Feedback, CallHistory
from .models import PrivateMessage
from django.shortcuts import render
from django.contrib.auth.models import User
from django.contrib import messages
import json
import random
import pickle
import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize

model = tf.keras.models.load_model("chatbot_model.h5")
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)
with open("encoder.pkl", "rb") as file:
    encoder = pickle.load(file)
with open('intents.json', 'r', encoding='utf-8') as file:
    data = json.load(file)


@csrf_exempt
def save_feedback(request):
    print("Incoming feedback request:", request.body)
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            print("Parsed JSON data:", data)
            x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
            ip_address = x_forwarded_for.split(',')[0] if x_forwarded_for else request.META.get('REMOTE_ADDR')
            required_fields = ['message_id', 'message', 'feedback']
            if not all(field in data for field in required_fields):
                return JsonResponse({'success': False, 'error': 'Missing required fields'}, status=400)
            feedback_data = {
                'message_id': data['message_id'],
                'message': data['message'],
                'feedback_type': data['feedback'],
                'context': data.get('conversation_context', []),  # Changed to match frontend
                'ip_address': ip_address,
            }

            if request.user.is_authenticated:
                feedback_data['user'] = request.user

            Feedback.objects.create(**feedback_data)
            return JsonResponse({'success': True})
        except json.JSONDecodeError:
            return JsonResponse({'success': False, 'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=400)
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def get_alternative_response(request):
    if request.method == 'GET':
        original_message = request.GET.get('message', '')
        original_message_id = request.GET.get('original_message_id', '')
        original_response = request.GET.get('original_response', '')        
        tag = predict_class(original_message, confidence_threshold=0.6)
        if tag:
            response_list = next((intent["response"] for intent in data["intents"] if intent["tag"] == tag), [])
            if len(response_list) > 1:
                alternative_responses = [r for r in response_list if r != original_response]
                if alternative_responses:
                    response = random.choice(alternative_responses)
                else:
                    response = f"Regarding '{original_message}', could you be more specific about what you're looking for?"
            else:
                response = f"About '{original_message}', I can also tell you more details if you'd like."
        else:
            response = f"I want to help with '{original_message}'. Could you rephrase or ask a different question?"
        
        return JsonResponse({
            'response': response,
            'original_message_id': original_message_id,
            'is_alternative': True
        })
    return JsonResponse({'error': 'Invalid request method'}, status=405)

def predict_class(text, confidence_threshold=0.7):
    if not text.strip():
        return None
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word in tokenizer.word_index]
    if not filtered_tokens:
        return None
    seq = tokenizer.texts_to_sequences([filtered_tokens])
    seq = pad_sequences(seq, maxlen=model.input_shape[1], padding="post")
    prediction = model.predict(seq)[0]
    predicted_index = np.argmax(prediction)
    confidence = prediction[predicted_index]
    if confidence < confidence_threshold:
        return None
    return encoder.classes_[predicted_index]

def chatbot_response(request):
    if request.method == "GET":
        user_message = request.GET.get("message", "").strip()
        if not user_message:
            return JsonResponse({"response": "🌟 Hello! How can I assist you today?"})        
        tag = predict_class(user_message)
        if tag:
            response_list = next((intent["response"] for intent in data["intents"] if intent["tag"] == tag), [])
            response = random.choice(response_list) if response_list else "🤔 I'm not sure I understand. Can you rephrase?"
        else:
            feedback_triggers = ["feedback", "suggest", "complain", "report", "experience"]
            if any(trigger in user_message.lower() for trigger in feedback_triggers):
                response = "💬 Thank you for your feedback! Could you tell me more about your experience?"
            else:
                response = "🔍 I'm still learning. Could you provide more details?"
        
        return JsonResponse({
            "response": response,
            "original_response": response  
        })
    return JsonResponse({"response": "Invalid request method. Use GET."})

def chatbot(request):
    return render(request, 'chatbot.html')

@receiver(user_logged_in)
def set_online(sender, request, user, **kwargs):
    user.is_online = True
    user.save()

@receiver(user_logged_out)
def set_offline(sender, request, user, **kwargs):
    user.is_online = False
    user.save()

def send_message(sender, recipient, message_text):
    message = PrivateMessage.objects.create(sender=sender, recipient=recipient, text=message_text, is_read=False)
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        f"notifications_{recipient.username}",
        {
            "type": "notify_new_message",
        }
    )

    return message


class LoginRedirectView(LoginRequiredMixin, View):
    def get(self, request):
        return redirect('users')


def login_view(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('users')
            else:
                form.add_error(None, 'Invalid Credentials')
        return render(request, 'chat/login.html', {'form': form, 'error': 'Invalid Credentials or CAPTCHA'})
    form = LoginForm()
    return render(request, 'chat/login.html', {'form': form})

def signup_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Registration successful! You can log in now.")
            return redirect('login')
        else:
            messages.error(request, "Error in registration. Please check the form.")
    else:
        form = UserCreationForm()

    return render(request, 'chat/signup.html', {'form': form})

@csrf_exempt
def save_message(request):
    if request.method == 'POST':
        message_content = request.POST.get('message')
        username = request.POST.get('username')
        try:
            user = User.objects.get(username=username)
            message = PrivateMessage.objects.create(
                recipient=user, sender=request.user, text=message_content
            )
            return JsonResponse({'status': 'success', 'message_id': message.id}, status=200)
        except User.DoesNotExist:
            return JsonResponse({'error': 'User not found'}, status=404)
    return JsonResponse({'error': 'Invalid request'}, status=400)


@login_required
def user_list_view(request):
    users = User.objects.exclude(id=request.user.id)
    return render(request, 'chat/users.html', {'users': users})

@login_required
def chat_view(request, username):
    user = request.user
    recipient = get_object_or_404(User, username=username)
    PrivateMessage.objects.filter(sender=request.user, recipient=recipient, is_read=False).update(is_read=True)
    messages = PrivateMessage.objects.filter(
        Q(sender=user, recipient=recipient) | Q(sender=recipient, recipient=user)
    ).order_by('timestamp')
    print("Messages Retrieved:", messages)
    return render(request, 'chat/users.html', {
        'messages': messages,
        'recipient': recipient,
        'user': user,
    })

def screenshare_view(request):
    return render(request, 'chat/users.html')

def users_view(request):
    return render(request, 'chat/users.html')

def custom_500_error(request):
    return render(request, '500.html', status=500)

@login_required
def fetch_new_messages(request, username):
    user = request.user
    messages = PrivateMessage.objects.filter(
        Q(sender__username=username, recipient=user) |
        Q(sender=user, recipient__username=username)
    ).order_by('-timestamp')

    message_data = [{
        'sender': message.sender.username,
        'content': message.text,
        'timestamp': message.timestamp.strftime('%H:%M')
    } for message in messages]

    return JsonResponse({'messages': message_data})

@csrf_exempt
@login_required
def mark_as_read(request):
    if request.method == "POST":
        username = request.POST.get("username")
        user = request.user
        PrivateMessage.objects.filter(sender__username=username, recipient=user, is_read=False).update(is_read=True)
        return JsonResponse({"success": True})
    return JsonResponse({"error": "Invalid request"}, status=400)

@login_required
def screen_share(request):
    return render(request, 'users.html')

@login_required
def fetch_unread_count(request, username):
    user = request.user
    recipient = User.objects.filter(username=username).first()
    if not recipient:
        return JsonResponse({'error': 'User not found'}, status=404)
    unread_count = PrivateMessage.objects.filter(
        recipient=recipient,
        is_read=False
    ).count()
    return JsonResponse({'unread_count': unread_count})

@login_required
def get_old_messages(request, sender, recipient, username):
    user = request.user
    recipient = User.objects.filter(username=username).first()
    messages = PrivateMessage.objects.filter(
        (Q(sender__username=sender, recipient__username=recipient) |
         Q(sender__username=recipient, recipient__username=sender))
    ).order_by("timestamp")
    message_data = [
        {
            "id": msg.id,
            "sender": msg.sender.username,
            "content": msg.content,
            "timestamp": msg.timestamp.strftime("%H:%M")
        }
        for msg in messages
    ]
    return JsonResponse({"messages": message_data})

def load_messages(request, sender, recipient):
    messages = PrivateMessage.objects.filter(
        sender__username__in=[sender, recipient],
        recipient__username__in=[sender, recipient]
    ).order_by("timestamp")

    return JsonResponse({"messages": [
        {"sender": msg.sender.username, "content": msg.content, "timestamp": msg.timestamp.strftime("%H:%M")}
        for msg in messages
    ]})


def custom_logout_view(request):
    logout(request)
    return redirect('/')

def custom_404_view(request, exception):
    return render(request, 'chat/404.html', status=404)

def custom_csrf_failure_view(request, reason=""):
    return render(request, "403_csrf.html", status=403)

def custom_500_view(request, exception):
    return render(request, '500.html', status=500)

from django.views.decorators.http import require_POST


@require_POST
@csrf_exempt
@login_required
def log_missed_call(request):
    try:
        data = json.loads(request.body)
        caller = get_object_or_404(User, username=data['caller'])
        receiver = request.user
        recent_missed = CallHistory.objects.filter(
            caller=caller,
            receiver=receiver,
            status__in=['missed', 'timeout', 'offline'],
            start_time__gte=timezone.now() - timedelta(seconds=5)
        ).exists()

        if recent_missed:
            return JsonResponse({'status': 'duplicate_ignored'})

        duplicate = CallHistory.objects.filter(
            caller=caller,
            receiver=receiver,
            status__in=['missed', 'timeout', 'offline'],
            start_time__gte=timezone.now() - timedelta(seconds=10)
        ).exists()

        if duplicate:
            return JsonResponse({'status': 'duplicate_ignored'})

        call = CallHistory.objects.create(
            caller=caller,
            receiver=receiver,
            call_type=data.get('call_type', 'video'),
            status=data.get('reason', 'missed'),
            start_time=timezone.now(),
            end_time=timezone.now()
        )

        return JsonResponse({'status': 'success'})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

@require_POST
@login_required
def log_rejected_call(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            if 'receiver' not in data:
                return JsonResponse({'error': 'Missing receiver'}, status=400)
            required_fields = ['receiver', 'call_type']
            for field in required_fields:
                if field not in data:
                    return JsonResponse(
                        {'error': f'Missing required field: {field}'},
                        status=400
                    )
            caller = get_object_or_404(User, username=data['caller'])
            receiver = request.user
            recent_rejected = CallHistory.objects.filter(
                caller=caller,
                receiver=receiver,
                status='rejected',
                start_time__gte=timezone.now() - timedelta(seconds=5)
            ).exists()

            if recent_rejected:
                return JsonResponse({'status': 'duplicate_ignored'})

            CallHistory.objects.create(
                caller=caller,
                receiver=receiver,
                call_type=data.get('call_type', 'video'),
                status='rejected',
                start_time=timezone.now()
            )

            return JsonResponse({'status': 'success'})
        except User.DoesNotExist:
            return JsonResponse({'error': 'Receiver user not found'}, status=404)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)


@require_POST
@csrf_exempt
@login_required
def log_completed_call(request):
    try:
        data = json.loads(request.body)
        required_fields = ['caller', 'receiver', 'call_type', 'duration']
        for field in required_fields:
            if field not in data:
                return JsonResponse({'error': f'Missing required field: {field}'}, status=400)

        # Validate duration is a number
        try:
            duration = float(data['duration'])
            if duration < 0:
                raise ValueError("Duration cannot be negative")
        except (ValueError, TypeError):
            return JsonResponse({'error': 'Invalid duration value'}, status=400)

        # Get users
        caller = User.objects.get(username=data['caller'])
        receiver = User.objects.get(username=data['receiver'])

        # Create call record
        CallHistory.objects.create(
            caller=caller,
            receiver=receiver,
            call_type=data['call_type'],
            status='completed',
            start_time=timezone.now() - timedelta(seconds=duration),
            end_time=timezone.now(),
            duration=timedelta(seconds=duration)
        )

        return JsonResponse({'status': 'success'})
    except User.DoesNotExist:
        return JsonResponse({'error': 'User not found'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

@login_required
def get_call_history(request):
    calls = CallHistory.objects.filter(
        (models.Q(caller=request.user) | models.Q(receiver=request.user)) &
        ~models.Q(caller=request.user, receiver=request.user)
    ).order_by('-start_time')[:50]
    history = []
    for call in calls:
        if call.status == 'rejected' and 'completed' in call.status.lower():
            call.status = 'rejected'
            call.save()

    for call in calls:
        history.append({
            'id': call.id,
            'call_type': call.call_type,
            'status': call.status,
            'timestamp': call.start_time.isoformat(),
            'duration': call.duration.total_seconds() if call.duration else None,
            'direction': 'outgoing' if call.caller == request.user else 'incoming',
            'other_user': call.receiver.username if call.caller == request.user else call.caller.username,
        })

    return JsonResponse({'history': history})


@require_POST
@csrf_exempt
@login_required
def log_call(request):
    try:
        data = json.loads(request.body)
        caller = request.user
        receiver = get_object_or_404(User, username=data['receiver'])

        # Determine direction automatically
        direction = 'outgoing' if data.get('is_initiator', False) else 'incoming'

        # Check for duplicates (5-second window)
        duplicate = CallHistory.objects.filter(
            caller=caller,
            receiver=receiver,
            start_time__gte=timezone.now() - timedelta(seconds=5)
        ).exists()

        if duplicate:
            return JsonResponse({'status': 'duplicate_ignored'})

        call = CallHistory.objects.create(
            caller=caller,
            receiver=receiver,
            call_type=data.get('call_type', 'video'),
            status=data.get('status', 'completed'),
            direction=direction,
            start_time=timezone.now(),
            end_time=timezone.now() if data.get('status') == 'completed' else None
        )

        return JsonResponse({
            'status': 'success',
            'call_id': call.id,
            'direction': call.direction
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)