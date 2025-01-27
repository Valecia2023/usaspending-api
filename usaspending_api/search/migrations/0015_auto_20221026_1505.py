# Generated by Django 3.2.15 on 2022-10-26 15:05

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('search', '0014_additional_transaction_search_cols'),
    ]

    operations = [
        migrations.AddField(
            model_name='awardsearch',
            name='base_and_all_options_value',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=23, null=True),
        ),
        migrations.AddField(
            model_name='awardsearch',
            name='base_exercised_options_val',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=23, null=True),
        ),
        migrations.AddField(
            model_name='awardsearch',
            name='certified_date',
            field=models.DateField(null=True),
        ),
        migrations.AddField(
            model_name='awardsearch',
            name='create_date',
            field=models.DateTimeField(null=True),
        ),
        migrations.AddField(
            model_name='awardsearch',
            name='earliest_transaction_id',
            field=models.IntegerField(db_index=True, null=True),
        ),
        migrations.AddField(
            model_name='awardsearch',
            name='fpds_agency_id',
            field=models.TextField(null=True),
        ),
        migrations.AddField(
            model_name='awardsearch',
            name='fpds_parent_agency_id',
            field=models.TextField(null=True),
        ),
        migrations.AddField(
            model_name='awardsearch',
            name='is_fpds',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='awardsearch',
            name='latest_transaction_id',
            field=models.IntegerField(db_index=True, null=True),
        ),
        migrations.AddField(
            model_name='awardsearch',
            name='non_federal_funding_amount',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=23, null=True),
        ),
        migrations.AddField(
            model_name='awardsearch',
            name='officer_1_amount',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=23, null=True),
        ),
        migrations.AddField(
            model_name='awardsearch',
            name='officer_1_name',
            field=models.TextField(null=True),
        ),
        migrations.AddField(
            model_name='awardsearch',
            name='officer_2_amount',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=23, null=True),
        ),
        migrations.AddField(
            model_name='awardsearch',
            name='officer_2_name',
            field=models.TextField(null=True),
        ),
        migrations.AddField(
            model_name='awardsearch',
            name='officer_3_amount',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=23, null=True),
        ),
        migrations.AddField(
            model_name='awardsearch',
            name='officer_3_name',
            field=models.TextField(null=True),
        ),
        migrations.AddField(
            model_name='awardsearch',
            name='officer_4_amount',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=23, null=True),
        ),
        migrations.AddField(
            model_name='awardsearch',
            name='officer_4_name',
            field=models.TextField(null=True),
        ),
        migrations.AddField(
            model_name='awardsearch',
            name='officer_5_amount',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=23, null=True),
        ),
        migrations.AddField(
            model_name='awardsearch',
            name='officer_5_name',
            field=models.TextField(null=True),
        ),
        migrations.AddField(
            model_name='awardsearch',
            name='parent_award_piid',
            field=models.TextField(db_index=True, null=True),
        ),
        migrations.AddField(
            model_name='awardsearch',
            name='raw_recipient_name',
            field=models.TextField(null=True),
        ),
        migrations.AddField(
            model_name='awardsearch',
            name='subaward_count',
            field=models.IntegerField(null=True),
        ),
        migrations.AddField(
            model_name='awardsearch',
            name='total_funding_amount',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=23, null=True),
        ),
        migrations.AddField(
            model_name='awardsearch',
            name='total_indirect_federal_sharing',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=23, null=True),
        ),
        migrations.AddField(
            model_name='awardsearch',
            name='total_subaward_amount',
            field=models.DecimalField(blank=True, decimal_places=2, max_digits=23, null=True),
        ),
        migrations.AddField(
            model_name='awardsearch',
            name='transaction_unique_id',
            field=models.TextField(null=True),
        ),
        migrations.AlterField(
            model_name='awardsearch',
            name='awarding_agency_id',
            field=models.IntegerField(db_index=True, null=True),
        ),
        migrations.AlterField(
            model_name='awardsearch',
            name='category',
            field=models.TextField(db_index=True, null=True),
        ),
        migrations.AlterField(
            model_name='awardsearch',
            name='fain',
            field=models.TextField(db_index=True, null=True),
        ),
        migrations.AlterField(
            model_name='awardsearch',
            name='funding_agency_id',
            field=models.IntegerField(db_index=True, null=True),
        ),
        migrations.AlterField(
            model_name='awardsearch',
            name='period_of_performance_current_end_date',
            field=models.DateField(db_index=True, null=True),
        ),
        migrations.AlterField(
            model_name='awardsearch',
            name='period_of_performance_start_date',
            field=models.DateField(db_index=True, null=True),
        ),
        migrations.AlterField(
            model_name='awardsearch',
            name='piid',
            field=models.TextField(db_index=True, null=True),
        ),
        migrations.AlterField(
            model_name='awardsearch',
            name='total_obligation',
            field=models.DecimalField(blank=True, db_index=True, decimal_places=2, max_digits=23, null=True),
        ),
        migrations.AlterField(
            model_name='awardsearch',
            name='type',
            field=models.TextField(db_index=True, null=True),
        ),
        migrations.AlterField(
            model_name='awardsearch',
            name='uri',
            field=models.TextField(db_index=True, null=True),
        ),
    ]
