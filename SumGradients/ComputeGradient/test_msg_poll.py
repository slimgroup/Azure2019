# Test getting messages from queue

from azure.storage.queue import QueueService
import json, base64

queue_service = QueueService(account_name='gatechsponsored', account_key='fHlppOUxjbINeTQDPpaLtUvJte/rnFUOtTPQ2tgJTMHwrpfAHQ9wYcz7Uiqjn177MpoIWqJxYVWjWL6bfkdeZw==')


messages = queue_service.get_messages('gradientqueuein', visibility_timeout=4, num_messages=2)
for message in messages:
    print(message.content)
    #queue_service.delete_message('taskqueue', message.id, message.pop_receipt)
