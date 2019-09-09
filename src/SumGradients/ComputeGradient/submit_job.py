import numpy as np
import json, base64, time
from azure.storage.queue import QueueService, QueueMessageFormat

########################################################################################################################
# Credentials

queue_service = QueueService(account_name='gatechsponsored', account_key='fHlppOUxjbINeTQDPpaLtUvJte/rnFUOtTPQ2tgJTMHwrpfAHQ9wYcz7Uiqjn177MpoIWqJxYVWjWL6bfkdeZw==')
queue_service.encode_function = QueueMessageFormat.text_base64encode

########################################################################################################################

# Job parameters
container = 'seismic'
partial_gradient_path = 'partial_gradients/'
full_gradient_path = 'full_gradients/'
gradient_name = 'test_grad_'
iteration = 1
maxiter = 3
batchsize = 100

# Encode msg and submit job
msg = container + '&' + partial_gradient_path + '&' + full_gradient_path + '&' + gradient_name + '&' + str(iteration) + '&' + str(maxiter) + '&' + str(batchsize)
queue_service.put_message('iterationqueue', msg)
