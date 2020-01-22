from azure.storage.blob import BlockBlobService, PublicAccess
import time

blob_service = BlockBlobService(account_name='', account_key='')

container = 'overthrust'
finished_shots = []

while True:

    # Get current shots
    current_blob_shot = blob_service.list_blobs(container, prefix='data/overthrust_3D_born_data_source_')
    num_items_shot = len(current_blob_shot.items)

    # Loop over blobs
    for i in range(num_items_shot):

        # Get shot index
        blob_name_shot = current_blob_shot.items[i].name
        shot_no = int(blob_name_shot[36:-12])

        # If not downloaded yet, check if all ranks are done
        if shot_no not in finished_shots:

            # Check if all ranks are present
            current_blob_shot = blob_service.list_blobs(container, prefix='data/overthrust_3D_born_data_source_' + str(shot_no) + 'rank_')
            num_items_rank = len(current_blob_shot.items)

            if num_items_rank == 4:
                for i in range(num_items_rank):



                    file_path_shot = '/data/pwitte3/azure_results/data/' + blob_name_shot[5:]

                    # Download result
                    tstart = time.time()
                    blob_service.get_blob_to_path(container, blob_name_rtm, file_path_rtm)
                    tend = time.time()

                    print('Download shot no: ', shot_no, ' in ', tend - tstart, ' seconds.')
                    finished_rtm.append(shot_no)

    time.sleep(1)
