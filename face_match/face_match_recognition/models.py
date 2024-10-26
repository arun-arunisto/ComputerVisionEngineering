from django.db import models

# Create your models here.
class UsersImages(models.Model):
    name = models.CharField(max_length=100)
    user_id = models.CharField(max_length=100)
    mobile = models.CharField(max_length=100)
    description = models.CharField(max_length=100)
    image = models.ImageField(upload_to='images')

    def __str__(self):
        return self.name