# Generated by Django 5.1.5 on 2025-05-16 06:35

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0016_userstatus_last_seen'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='userstatus',
            name='last_seen',
        ),
    ]
