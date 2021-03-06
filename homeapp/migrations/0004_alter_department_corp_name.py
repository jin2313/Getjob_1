# Generated by Django 3.2.13 on 2022-05-09 13:54

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('homeapp', '0003_rename_corp_id_department_corp_name'),
    ]

    operations = [
        migrations.AlterField(
            model_name='department',
            name='corp_name',
            field=models.ForeignKey(db_column='corp_name', on_delete=django.db.models.deletion.CASCADE, related_name='corp', to='homeapp.corporation', to_field='name'),
        ),
    ]
