import urllib3
import json
import traceback
    
webhook_url = 'https://hooks.slack.com/services/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'
 
def notify(message):
    try:
        slack_message = {'text':message}
        http = urllib3.PoolManager()
        response = http.request('POST',
            webhook_url,
            body = json.dumps(slack_message),
            headers={'Content-Type':'application/json'},
            retries = False)
    except:
        traceback.print_exc()

    return True
