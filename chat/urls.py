from captcha.views import captcha_refresh
from django.urls import path
from . import views
from .views import users_view, LoginRedirectView, mark_as_read, load_messages, chatbot_response, custom_404_view, custom_500_view, custom_csrf_failure_view
from django.contrib.auth import views as auth_views
from .views import custom_logout_view
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('log_call/', views.log_call, name='log_call'),
    path('get_call_history/', views.get_call_history, name='get_call_history'),
    path('log_missed_call/', views.log_missed_call, name='log_missed_call'),
    path('log_rejected_call/', views.log_rejected_call, name='log_rejected_call'),
    path('log_completed_call/', views.log_completed_call, name='log_completed_call'),
    path('refresh-captcha/', captcha_refresh, name='captcha-refresh'),
    path('get-response/', views.chatbot_response, name='chatbot_response'),
    path('save-feedback/', views.save_feedback, name='save_feedback'),
    path('get-alternative-response/', views.get_alternative_response, name='get_alternative_response'),
    path('save-feedback/', views.save_feedback, name='save_feedback'),
    path('get-response/', chatbot_response, name="chatbot_response"),
    path('users/', views.user_list_view, name='user_list'),
    path('chat/<str:username>/', views.chat_view, name='chat'),
    path('mark-as-read/', mark_as_read, name='mark_as_read'),
    path('load-messages/<str:sender>/<str:recipient>/', load_messages, name='load-messages'),
    path('', views.login_view, name='login'),
    path('signup/', views.signup_view, name='signup'),
    path('users/', users_view, name='users'),
    path('chat/fetch-messages/<str:username>/', views.fetch_new_messages, name='fetch_new_messages'),
    path('login_redirect/', LoginRedirectView.as_view(), name='login_redirect'),
    path('screenshare/<str:room_name>/', views.screen_share, name='screen_share'),
    path('password-reset/', auth_views.PasswordResetView.as_view(), name='password_reset'),
    path('password-reset/done/', auth_views.PasswordResetDoneView.as_view(), name='password_reset_done'),
    path('password-reset-confirm/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(), name='password_reset_confirm'),
    path('password-reset-complete/', auth_views.PasswordResetCompleteView.as_view(), name='password_reset_complete'),
    path('logout/', custom_logout_view, name='logout'),
    path('chatbot/', views.chatbot, name='chatbot')
]
handler = custom_404_view
handler403 = custom_csrf_failure_view
handler500 = custom_500_view

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)