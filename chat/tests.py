from django.test import TestCase
from django.urls import reverse
from django.contrib.auth.models import User
from channels.testing import WebsocketCommunicator
from chat.routing import websocket_urlpatterns
import json

class UserListViewTests(TestCase):
    def setUp(self):
        # Create a test user
        self.user = User.objects.create_user(username='Iyanna', password='jene2003')

    def test_user_list_page_renders(self):
        # Test if the user list page renders correctly
        self.client.login(username='Iyanna', password='jene2003')
        response = self.client.get(reverse('user_list'))  # Replace 'user_list' with your URL name
        self.assertTemplateUsed(response, 'chat/users.html')  # Replace with your template name

class DynamicContentTests(TestCase):
    def setUp(self):
        self.user1 = User.objects.create_user(username='user01', password='jene2003')
        self.user2 = User.objects.create_user(username='user02', password='jene2003')

    def test_user_list_display(self):
        self.client.login(username='user01', password='jene2003')
        response = self.client.get(reverse('user_list'))
        self.assertContains(response, 'user01')
        self.assertContains(response, 'user02')

    def test_chat_functionality(self):
        # Test chat functionality
        self.client.login(username='user01', password='jene2003')

class LogoutTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='Iyanna', password='jene2003')

    def test_logout(self):
        self.client.login(username='Iyanna', password='jene2003')
        response = self.client.get(reverse('logout'))
        self.assertRedirects(response, reverse('login'))