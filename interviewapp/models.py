from django.db import models

# Create your models here.
class Question(models.Model):
    quest_id = models.IntegerField(null=False, unique=True, default=0)
    level = models.IntegerField(null=False)
    content = models.TextField(null=False)

    def __str__(self):
        return str(self.quest_id)