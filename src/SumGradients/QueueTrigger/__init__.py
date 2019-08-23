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


def get_random_name(length):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(length))


def get_multipart_file_params(container, blob_name):

    meta = blob_service.get_blob_properties(container, blob_name)
    num_bytes = meta.properties.content_length
    min_bytes = 128*1024**2   # minimum number of bytes for first object in multi-part object
    desired_part_size = 512 * 1024**2    # want part size of 512 MB

    # Determine number or parts and size of final part
    if num_bytes <= min_bytes or num_bytes <= desired_part_size:
        num_parts = 1
        residual_bytes = num_bytes
    else:
        num_parts = int(num_bytes/desired_part_size)
        if num_bytes % desired_part_size > 0:
            num_parts += 1
            residual_bytes = num_bytes % desired_part_size
        else:
            residual_bytes = None
    return num_parts, desired_part_size, residual_bytes, num_bytes




########################################################################################################################


def main(queuemsg: func.QueueMessage, msg: func.Out[func.QueueMessage]):

    # Extract message'
    msg1 = extract_message(queuemsg)
    count, batchsize = extract_parameters(msg1)[4:]

    if count < batchsize:
        try:

            # Get second message
            msg2_b64 = queue_service.get_messages('gradientqueuein', visibility_timeout=10, num_messages=1)
            print('Found ', len(msg2_b64), ' extra message(s).')
            msg2 = base64.b64decode(msg2_b64[0].content).decode()
            queue_service.delete_message('gradientqueuein', msg2_b64[0].id, msg2_b64[0].pop_receipt)

            # Get gradient parameters
            container1, partial_path1, grad_name1, idx1, count1, batchsize1 = extract_parameters(msg1)
            container2, partial_path2, grad_name2, idx2, count2, batchsize2 = extract_parameters(msg2)

            # New block blob gradient
            get_multipart_file_params(container1, partial_path1 + grad_name1 + idx1)
            idx3 = get_random_name(16)
            blob_name = partial_path2 + grad_name2 + idx3

            # Loop over blocks
            byte_count = 0
            blocks = []
            for part in range(num_parts):

                # Get byte range
                byte_start = byte_count  # byte start
                if residual_bytes is not None and part == (num_parts-1):
                    byte_end = byte_count + residual_bytes - 1
                else:
                    byte_end = byte_count + desired_part_size - 1 # read until end of blob

                # Get current gradients and sum
                g = array_get(container1, partial_path1 + grad_name1 + idx1, start_range=byte_start, end_range=byte_end)

                g += array_get(container2, partial_path2 + grad_name2 + idx2, start_range=byte_start, end_range=byte_end)

                # Write back to blob storage
                block_id = get_random_name(32)
                blob_service.put_block(container1, blob_name, g.tostring(), block_id)
                blocks.append(BlobBlock(id=block_id))

            # Finalize block blob and send message to queue
            blob_service.put_block_list(container, blob_name, blocks)

            # Out message
            msg_out = container1 + '&' + partial_path2 + '&' + str(idx3) + '&' + str(count1 + count2) + '&' + str(batchsize1)
            print('Out message: ', msg_out, '\n')
            msg.set(msg_out)

        except:
            print('No other messages found. Return message to queue.')
            #time.sleep(2)
            msg.set(msg1)
    else:
        print("Update variable and done with reduction.\n")
