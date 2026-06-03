"""
WSGI config for deepfake_web project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/wsgi/
"""

import os

# Disable CUDA and eager loading before importing torch (memory optimization)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'deepfake_web.settings')

application = get_wsgi_application()
