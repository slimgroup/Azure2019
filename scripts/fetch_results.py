from azure.storage.blob import BlockBlobService, PublicAccess
import time

blob_service = BlockBlobService(account_name='', account_key='')

container = 'overthrust'
finished_rtm = []
finished_timings = []

while True:

    # Get current rtm images
    current_blob_rtm = blob_service.list_blobs(container, prefix='rtm/overthrust_3D_rtm_shot_')
    num_items_rtm = len(current_blob_rtm.items)

    # Loop over blobs
    for i in range(num_items_rtm):

        # Get shot index
        blob_name_rtm = current_blob_rtm.items[i].name
        shot_no = int(blob_name_rtm[27:-3])

        if shot_no not in finished_rtm:
            file_path_rtm = '/data/pwitte3/azure_results/rtm/' + blob_name_rtm[4:]

            # Download result
            tstart = time.time()
            blob_service.get_blob_to_path(container, blob_name_rtm, file_path_rtm)
            tend = time.time()

            print('Download shot no: ', shot_no, ' in ', tend - tstart, ' seconds.')
            finished_rtm.append(shot_no)


    # Get current timings
    current_blob_timings = blob_service.list_blobs(container, prefix='timings/timings_rtm_3D_shot_')
    num_items_timings = len(current_blob_timings.items)

    # Loop over blobs
    for i in range(num_items_timings):

        # Get shot index
        blob_name_timings = current_blob_timings.items[i].name
        shot_no = int(blob_name_timings[28:-4])

        if shot_no not in finished_timings:
            file_path_timings = '/data/pwitte3/azure_results/timings/' + blob_name_timings[8:]

            # Download result
            tstart = time.time()
            blob_service.get_blob_to_path(container, blob_name_timings, file_path_timings)
            tend = time.time()

            print('Download timings no: ', shot_no, ' in ', tend - tstart, ' seconds.')
            finished_timings.append(shot_no)

    time.sleep(1)
