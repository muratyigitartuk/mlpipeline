import os
import json
import urllib.request

def send_slack_alert(message: str, webhook_url: str = None):
    url = webhook_url or os.getenv('SLACK_WEBHOOK')
    if not url:
        return False
    data = json.dumps({'text': message}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return 200 <= resp.getcode() < 300
    except Exception:
        return False
