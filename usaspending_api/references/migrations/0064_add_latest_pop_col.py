# Generated by Django 3.2.23 on 2023-12-14 23:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('references', '0063_add_enactment_date_to_defc'),
    ]

    operations = [
        migrations.AddField(
            model_name='refcountrycode',
            name='latest_population',
            field=models.BigIntegerField(null=True),
        ),
    ]