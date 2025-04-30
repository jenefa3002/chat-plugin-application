from django.contrib import admin
from chat.models import PrivateMessage, UserStatus, Feedback

admin.site.register(PrivateMessage)
admin.site.register(UserStatus)
admin.site.register(Feedback)