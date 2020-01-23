import numpy as np
from AzureUtilities import array_put
import json, base64, time
from azure.storage.queue import QueueService, QueueMessageFormat

queue_service = QueueService(account_name='', account_key='')
queue_service.encode_function = QueueMessageFormat.text_base64encode

batchsize = 100
shape = int(500)    # 3.8 MB

for j in range(batchsize):
    g = np.array(np.linspace(1, shape, shape)).astype('float32')
    array_put(g, 'seismic', 'partial_gradients/test_grad_' + str(j+1))

print('Done uploading')
#time.sleep(10)

for j in range(batchsize):
    print('Send message ', j+1, ' of ', batchsize)
    time.sleep(np.random.rand(1))
    idx = str(j+1)
    msg = 'seismic&partial_gradients/&full_gradients/&test_grad_&' + idx + '&1&3&1&' + str(batchsize)
    queue_service.put_message('gradientqueue', msg)
