import logging
import azure.functions as func
import numpy as np
import json, base64, time, random, string
from azure.storage.blob import BlockBlobService, PublicAccess
from azure.storage.blob.models import BlobBlock
from azure.storage.queue import QueueService, QueueMessageFormat


########################################################################################################################
# Credentials

blob_service = BlockBlobService(account_name='', account_key='')

queue_service = QueueService(account_name='', account_key='')

queue_service.encode_function = QueueMessageFormat.text_base64encode

#######################################################################################################################


def convert_to_string(t):
    if len(t) == 1:
        return str(t[0])
    elif len(t) == 2:
        return str(t[0]) + 'S' + str(t[1])
    else:
        return str(t[0]) + 'S' + str(t[1]) + 'S' + str(t[2])

def convert_int_from_string(s):
    s_split = s.split('S')
    ndim = len(s_split)
    if ndim==1:
        n = int(s_split[0])
    elif ndim==2:
        n1 = int(s_split[0])
        n2 = int(s_split[1])
        n = (n1, n2)
    else:
        n1 = int(s_split[0])
        n2 = int(s_split[1])
        n3 = int(s_split[2])
        n = (n1, n2, n3)
    return n

def convert_float_from_string(s):
    s_split = s.split('S')
    ndim = len(s_split)
    d1 = float(s_split[0])
    d2 = float(s_split[1])
    if ndim==2:
        d = (d1, d2)
    else:
        d3 = float(s_split[2])
        d = (d1, d2, d3)
    return d

# write array
def array_put(blob, container, blob_name, index=0, count=None, validate_content=False):
    shape_str = convert_to_string(blob.shape)
    meta = {'dtype':str(blob.dtype), 'shape': shape_str}
    blob_service.create_blob_from_bytes(
        container,
        blob_name,
        blob.tostring(),   # blob
        index = index,    # start index in array of bytes
        count = count, # number of bytes to upload
        metadata = meta,  # Name-value pairs
        validate_content = validate_content
    )

# put array
def array_get(container, blob_name, start_range=None, end_range=None, validate_content=False):
    binary_blob = blob_service.get_blob_to_bytes(
        container,
        blob_name,
        start_range=start_range,
        end_range=end_range,
        validate_content=validate_content
    )
    try:
        meta = binary_blob.metadata
        shape = convert_int_from_string(meta['shape'])
        x = np.fromstring(binary_blob.content, dtype=meta['dtype'])
        return x.reshape(shape)
    except:
        x = np.fromstring(binary_blob.content, dtype='float32')
        return x

def extract_message(queuemsg):
    msg = queuemsg.get_body().decode('utf-8')
    return msg

def extract_parameters(msg):
    msg_list = msg.split('&')
    print("len of message: ", len(msg_list))

    # Extract parameters
    container = msg_list[0]
    partial_path = msg_list[1]
    full_path = msg_list[2]
    grad_name = msg_list[3]
    iteration = int(msg_list[4])
    maxiter = int(msg_list[5])
    batchsize = int(msg_list[6])
    #chunk = msg_list[8]
    #queue_name = msg_list[5]
    #variable_path = msg_list[10]
    #variable_name = msg_list[11]
    #step_length = int(msg_list[12])
    #step_scaling = int(msg_list[13])

    return container, partial_path, full_path, grad_name, iteration, maxiter, batchsize


########################################################################################################################


# Launch Azure Batch job
def main(queuemsg: func.QueueMessage) -> None:

    # Extract message
    msg = extract_message(queuemsg)
    container, partial_path, full_path, grad_name, iter, maxiter, batchsize = extract_parameters(msg)
    print('Iteration ', iter, ' of ', maxiter)

    # Termination criterion (e.g. iteration number)
    if iter <= maxiter:

        # Start Azure Batch job

        # Dummy example: create artificial gradients and send to queue
        shape = int(1e3)    # 3.8 MB

        for j in range(batchsize):
            g = np.array(np.linspace(1, shape, shape)).astype('float32')
            array_put(g, container, partial_path + grad_name + str(j+1))

        for j in range(batchsize):
            time.sleep(np.random.rand(1))
            idx = str(j+1)
            count = 1
            msg_out = container + '&' + partial_path + '&' + full_path + '&' + grad_name + '&' + str(j+1) + '&' + str(iter) + '&' + str(maxiter) + '&' + str(count) + '&' + str(batchsize)
            print('Out message: ', msg_out, '\n')
            queue_service.put_message('gradientqueue', msg_out)

    else:
        # Done
        print('Sucessfully completed workflow after ', maxiter, ' iteration.')
