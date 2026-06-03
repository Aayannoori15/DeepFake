web: gunicorn deepfake_web.wsgi:application --workers 1 --worker-class sync --worker-tmp-dir /dev/shm --max-requests 100 --max-requests-jitter 50 --timeout 60 --graceful-timeout 30
