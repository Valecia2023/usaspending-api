# Generated by Django 3.2.18 on 2023-03-02 21:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("search", "0030_force_as_linkage_to_ts"),
    ]

    operations = [
        migrations.AddField(
            model_name="awardsearch",
            name="iija_spending_by_defc",
            field=models.JSONField(null=True),
        ),
        migrations.AddField(
            model_name="awardsearch",
            name="total_iija_obligation",
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=23, null=True),
        ),
        migrations.AddField(
            model_name="awardsearch",
            name="total_iija_outlay",
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=23, null=True),
        ),
    ]
