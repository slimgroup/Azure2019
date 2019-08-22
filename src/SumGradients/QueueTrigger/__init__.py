import logging
import azure.functions as func
import numpy as np
import json, base64, time
from azure.storage.blob import BlockBlobService, PublicAccess
from azure.storage.queue import QueueService


########################################################################################################################
# Credentials

blob_service = BlockBlobService(account_name='', account_key='')

queue_service = QueueService(account_name='', account_key='')


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
    meta = binary_blob.metadata
    shape = convert_int_from_string(meta['shape'])
    x = np.fromstring(binary_blob.content, dtype=meta['dtype'])
    return x.reshape(shape)


def extract_message(queuemsg):
    msg = queuemsg.get_body().decode('utf-8')
    return msg


def extract_parameters(msg):
    msg_list = msg.split('&')
    #print('No of items in list: ', len(msg_list))

    # Extract parameters
    container = msg_list[0]
    partial_path = msg_list[1]
    #full_path = msg_list[2]
    grad_name = msg_list[2]
    idx = msg_list[3]
    #iteration = msg_list[5]
    count = int(msg_list[4])
    batchsize = int(msg_list[5])
    #chunk = msg_list[8]
    #queue_name = msg_list[5]
    #variable_path = msg_list[10]
    #variable_name = msg_list[11]
    #step_length = int(msg_list[12])
    #step_scaling = int(msg_list[13])

    return container, partial_path, grad_name, idx, count, batchsize
    #return bucket, partial_path, full_path, grad_name, idx, iteration, count, batchsize, \
    #    chunk, queue_name, variable_path, variable_name, step_length, step_scaling

########################################################################################################################


def main(queuemsg: func.QueueMessage, msg: func.Out[func.QueueMessage]):

    # Extract message'
    msg1 = extract_message(queuemsg)
    count, batchsize = extract_parameters(msg1)[4:]

    if count < batchsize:
        try:
            msg2_b64 = queue_service.get_messages('gradientqueuein', visibility_timeout=10, num_messages=1)
            print('Found ', len(msg2_b64), ' extra message(s).')
            msg2 = base64.b64decode(msg2_b64[0].content).decode()
            queue_service.delete_message('gradientqueuein', msg2_b64[0].id, msg2_b64[0].pop_receipt)

            # Load gradient
            container1, partial_path1, grad_name1, idx1, count1, batchsize1 = extract_parameters(msg1)
            container2, partial_path2, grad_name2, idx2, count2, batchsize2 = extract_parameters(msg2)

            # Sum gradients
            g1 = array_get(container1, partial_path1 + grad_name1 + idx1)
            g2 = array_get(container2, partial_path2 + grad_name2 + idx2)
            array_put(g1 + g2, container1, partial_path1 + grad_name1 + str(3))

            # Out message
            msg_out = 'seismic&partial_gradients/&test_grad_&3&2&2'
            msg.set(msg_out)

        except:
            print('No other messages found. Return message to queue.')
            #time.sleep(2)
            msg.set(msg1)

    else:
        print("Done with gradient reduction.\n")
