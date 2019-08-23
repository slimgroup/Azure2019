import numpy as np
from AzureUtilities import array_put
import json, base64, time
from azure.storage.queue import QueueService, QueueMessageFormat

queue_service = QueueService(account_name='', account_key='')
queue_service.encode_function = QueueMessageFormat.text_base64encode

g1 = np.ones(100, dtype='float32')
g2 = np.ones(100, dtype='float32')*2

array_put(g1, 'seismic', 'partial_gradients/test_grad_1')
msg1 = 'seismic&partial_gradients/&test_grad_&1&1&2'
queue_service.put_message('gradientqueuein', msg1)

time.sleep(4)

array_put(g2, 'seismic', 'partial_gradients/test_grad_2')
msg2 = 'seismic&partial_gradients/&test_grad_&2&1&2'
queue_service.put_message('gradientqueuein', msg2)
