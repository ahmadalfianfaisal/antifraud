import os
import subprocess

bind = f"0.0.0.0:{os.environ.get('PROXY_PORT', '8443')}"
workers = 4
worker_class = "gevent"
timeout = 60
keepalive = 5
certfile = "cert.pem"
keyfile = "key.pem"
accesslog = "-"
errorlog = "-"
