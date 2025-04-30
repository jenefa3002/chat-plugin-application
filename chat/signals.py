from django.contrib.auth.signals import user_logged_in, user_logged_out
from django.dispatch import receiver
from django.utils.timezone import now
from django.contrib.auth.models import User

@receiver(user_logged_in)
def set_online(sender, request, user, **kwargs):
    user.is_online = True
    user.save()

@receiver(user_logged_out)
def set_offline(sender, request, user, **kwargs):
    user.is_online = False
    user.save()

from django.contrib.auth.signals import user_logged_in, user_logged_out
from django.dispatch import receiver
from chat.models import UserStatus

@receiver(user_logged_in)
def set_online(sender, request, user, **kwargs):
    status, created = UserStatus.objects.get_or_create(user=user)
    status.is_online = True
    status.save()

@receiver(user_logged_out)
def set_offline(sender, request, user, **kwargs):
    status, created = UserStatus.objects.get_or_create(user=user)
    status.is_online = False
    status.save()
