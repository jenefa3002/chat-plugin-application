import json
from collections import defaultdict
from django.utils import timezone
from channels.db import database_sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer, AsyncJsonWebsocketConsumer
from asgiref.sync import sync_to_async
from django.contrib.auth.models import User
import logging
from .models import PrivateMessage, UserStatus

logger = logging.getLogger(__name__)

class PrivateChatConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        logger.info(f"Attempting WebSocket connection: {self.channel_name}")
        try:
            self.sender_username = self.scope['url_route']['kwargs']['sender_username']
            self.recipient_username = self.scope['url_route']['kwargs']['recipient_username']
            self.room_name = f"chat_{min(self.sender_username, self.recipient_username)}_{max(self.sender_username, self.recipient_username)}"
            self.room_group_name = f"chat_{self.room_name}"
            logger.info(f"Room Group: {self.room_group_name}")

            await self.channel_layer.group_add(self.room_group_name, self.channel_name)
            await self.accept()
            self.sender = await database_sync_to_async(User.objects.get)(username=self.sender_username)
            self.recipient = await database_sync_to_async(User.objects.get)(username=self.recipient_username)

            logger.info(f"WebSocket connected: {self.room_group_name}, Channel: {self.channel_name}")
        except Exception as e:
            logger.error(f"WebSocket connection error: {str(e)}")
            await self.close()

    async def disconnect(self, close_code):
        logger.info(f"WebSocket disconnected: {self.channel_name}")
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)

    async def receive(self, text_data):
        try:
            text_data_json = json.loads(text_data)
            message = text_data_json.get('message', None)

            if not message:
                await self.send(text_data=json.dumps({'error': 'No message provided'}))
                return
            if text_data_json.get('type') == 'typing':
                await self.channel_layer.group_send(
                    self.room_group_name,
                    {
                        'type': 'typing_indicator',
                        'sender': self.sender_username,
                        'recipient': text_data_json['recipient']
                    }
                )
            elif text_data_json.get('type') == 'stopped_typing':
                await self.channel_layer.group_send(
                    self.room_group_name,
                    {
                        'type': 'chat.stopped_typing',
                        'sender': self.sender_username,
                        'recipient': text_data_json['recipient']
                    }
                )
            else:
                message = text_data_json['message']

            new_message = await self.save_message(self.sender_username, self.recipient_username, message)

            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'chat_message',
                    'message': message,
                    'sender': self.sender_username,
                    'recipient': self.recipient_username,
                    'timestamp': new_message.timestamp.strftime("%H:%M"),
                }
            )

            logger.info(f"Message received: {message}")
        except json.JSONDecodeError:
            logger.error("Invalid JSON received")
            await self.send(text_data=json.dumps({'error': 'Invalid JSON'}))
        except User.DoesNotExist:
            await self.send(text_data=json.dumps({'error': 'User not found'}))

    async def get_recipient(self, username):
        return await sync_to_async(User.objects.get)(username=username)

    async def chat_typing(self, event):
        if self.recipient_username == event['recipient']:
            await self.send(text_data=json.dumps({
                'type': 'typing',
                'sender': event['sender']
            }))

    async def chat_stopped_typing(self, event):
        if self.recipient_username == event['recipient']:
            await self.send(text_data=json.dumps({
                'type': 'stopped_typing',
                'sender': event['sender']
            }))

    async def chat_message(self, event):
        await self.send(text_data=json.dumps({
            'type': 'chat_message',
            'message': event['message'],
            'sender': event['sender'],
            'recipient': event['recipient'],
            'timestamp': event['timestamp'],
        }))

    async def send_chat_history(self):
        messages = await self.chat_history()
        messages_data = [
            {
                "message": msg.content,
                "sender": msg.sender.username,
                "timestamp": msg.timestamp.strftime("%H:%M"),
            }
            for msg in messages
        ]

        await self.send(text_data=json.dumps({
            "type": "chat_history",
            "messages": messages_data
        }))

    @database_sync_to_async
    def chat_history(self):
        return PrivateMessage.objects.filter(
            sender__username__in=[self.sender_username, self.recipient_username],
            recipient__username__in=[self.sender_username, self.recipient_username]
        ).order_by('timestamp')

    @database_sync_to_async
    def save_message(self, sender_username, recipient_username, content):
        sender_user = User.objects.get(username=sender_username)
        recipient_user = User.objects.get(username=recipient_username)
        return PrivateMessage.objects.create(
            sender=sender_user,
            recipient=recipient_user,
            content=content,
        )

    async def user_online(self, event):
        await self.send(text_data=json.dumps({
            'type': 'user_status',
            'username': event['username'],
            'status': 'online',
        }))

    async def user_offline(self, event):
        await self.send(text_data=json.dumps({
            'type': 'user_status',
            'username': event['username'],
            'status': 'offline',
        }))

    async def send_notification(self, recipient_username, message):
        try:
            recipient = await database_sync_to_async(User.objects.get)(username=recipient_username)
            if recipient.is_authenticated:
                await self.channel_layer.group_send(
                    f"notifications_{recipient_username}",
                    {
                        'type': 'notification_message',
                        'message': message,
                        'sender': self.sender_username
                    }
                )
        except User.DoesNotExist:
            logger.error(f"Notification error: User {recipient_username} not found")


class NotificationConsumer(AsyncJsonWebsocketConsumer):

    async def connect(self):
        self.user = self.scope["user"]
        if self.user.is_authenticated:
            self.group_name = f"notifications_{self.user.username}"
            await self.channel_layer.group_add(self.group_name, self.channel_name)
            await self.accept()
            await self.send_unread_messages_per_sender()

    async def disconnect(self, close_code):
        if self.user.is_authenticated:
            await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def receive_json(self, content):
        sender_username = content.get("sender_username")

        if content.get("type") == "mark_read" and sender_username:
            await self.mark_messages_as_read(sender_username)
            await self.send_unread_messages_per_sender()

    @database_sync_to_async
    def mark_messages_as_read(self, sender_username):
        PrivateMessage.objects.filter(
            recipient=self.user,
            sender__username=sender_username,
            is_read=False
        ).update(is_read=True)

    @database_sync_to_async
    def get_unread_messages_per_sender(self):
        unread_messages = PrivateMessage.objects.filter(recipient=self.user, is_read=False)
        unread_counts = defaultdict(int)
        for message in unread_messages:
            unread_counts[message.sender.username] += 1
        return unread_counts

    async def send_unread_messages_per_sender(self):
        unread_counts = await self.get_unread_messages_per_sender()
        await self.send_json({"type": "unread_count", "unread_counts": unread_counts})

    async def notify_new_message(self, event):
        await self.send_unread_messages_per_sender()


class ScreenShareConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.sender_username = self.scope['url_route']['kwargs']['sender_username']
        self.recipient_username = self.scope['url_route']['kwargs']['recipient_username']
        self.room_group_name = f'screenshare_{min(self.sender_username, self.recipient_username)}_{max(self.sender_username, self.recipient_username)}'

        self.sender = await database_sync_to_async(User.objects.get)(username=self.sender_username)
        self.recipient = await database_sync_to_async(User.objects.get)(username=self.recipient_username)

        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name,
        )
        await self.accept()
        logger.info("Websocket connected")

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )
        logger.info("Websocket disconnected for Screen Shareing")

    async def receive(self, text_data):
        message = json.loads(text_data)
        logger.info("Reciving signal..")
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'signal_message',
                'message': message
            }
        )

    async def signal_message(self, event):
        message = event['message']
        await self.send(text_data=json.dumps(message))


class OnlineStatusConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        self.user = self.scope["user"]
        if self.user.is_authenticated:
            await self.set_online_status(self.user, True)
            await self.channel_layer.group_add("online_users", self.channel_name)
            await self.accept()
            await self.broadcast_all_users_status()

    async def disconnect(self, close_code):
        if self.user.is_authenticated:
            await self.set_online_status(self.user, False)
            await self.channel_layer.group_discard("online_users", self.channel_name)
            await self.broadcast_all_users_status()

    async def broadcast_all_users_status(self):
        online_users = await self.get_online_users()
        await self.channel_layer.group_send(
            "online_users",
            {
                "type": "update_online_users",
                "online_users": online_users
            }
        )

    async def update_online_users(self, event):
        await self.send(text_data=json.dumps(event))

    @database_sync_to_async
    def set_online_status(self, user, status):
        user_status, _ = UserStatus.objects.get_or_create(user=user)
        user_status.is_online = status
        if status:
            user_status.last_seen = timezone.now()
        user_status.save()

    @database_sync_to_async
    def get_online_users(self):
        return list(UserStatus.objects.filter(is_online=True).values("user__id", "user__username"))

class VideoCallConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.sender = self.scope['url_route']['kwargs']['sender']
        self.recipient = self.scope['url_route']['kwargs']['recipient']
        self.room_name = f"videocall_{min(self.sender, self.recipient)}_{max(self.sender, self.recipient)}"
        self.room_group_name = f"videocall_{self.room_name}"

        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        await self.accept()

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        data = json.loads(text_data)
        data['sender'] = self.sender

        # Send message to room group
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'videocall_message',
                'data': data
            }
        )

    async def videocall_message(self, event):
        data = event['data']
        await self.send(text_data=json.dumps(data))