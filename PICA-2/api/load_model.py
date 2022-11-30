from google.cloud import storage
import zipfile
import os

client = storage.Client()
bucket = client.bucket("model_checkpoints")
#blob = bucket.blob("apple")
#blob.download_to_filename(blob.name)

#blobs = client.list_blobs("model_checkpoints")
#for blob in blobs:
#    print(blob.name)


def extract_to(parent_blob):
    ##Unzips all files in our parent blob we just downloaded from GCS
    for zipped_file in os.listdir(parent_blob):
        print(f"Unzipping {zipped_file}...")
        with zipfile.ZipFile(os.path.join(parent_blob, zipped_file),
                             'r') as zip_ref:
            #we do not define an extraction directory since the zipped files already contain
            #the directories, since there was a large folder containing
            #all subsequent folders with files belonging to each category
            zip_ref.extractall()
        print("Success!")
        
with zipfile.ZipFile("apple",
                    'r') as zip_ref:
            #we do not define an extraction directory since the zipped files already contain
            #the directories, since there was a large folder containing
            #all subsequent folders with files belonging to each category
            zip_ref.extractall()