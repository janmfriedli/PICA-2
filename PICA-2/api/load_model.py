from google.cloud import storage
import zipfile
import os



names = ["house","apple","squirrel","mountain","door","cloud"]

central_dir = "central_models"
if not os.path.isdir(central_dir):
    os.mkdir(central_dir)

client = storage.Client()
bucket = client.bucket("model_checkpoints")

for name in names:
    print(f"Working on {name}...")
    blob = bucket.blob(name)
    blob.download_to_filename(blob.name)

    subdir_name = os.path.join(central_dir , name)
    if not os.path.isdir(subdir_name):
        os.mkdir(subdir_name)

    with zipfile.ZipFile(name, "r") as zip_ref:
            #we do not define an extraction directory since the zipped files already contain
            #the directories, since there was a large folder containing
            #all subsequent folders with files belonging to each category
            zip_ref.extractall(subdir_name)
    print(f"Finished zipping task {name}...")
    os.remove(name)
    print(f"Removed zipped {name}")

#client = storage.Client()
#bucket = client.bucket('model_checkpoints')
#blobs = client.list_blobs("model_checkpoints"):
    #for name in
#for blob in blobs:
#print(blob.name)




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






"""
if __name__ == '__main__':
    client = storage.Client()
    bucket = client.bucket("model_checkpoints")
    blob = bucket.blob(NAME)
    blob.download_to_filename(blob.name)

    with zipfile.ZipFile(NAME,
                    'r') as zip_ref:
            #we do not define an extraction directory since the zipped files already contain
            #the directories, since there was a large folder containing
            #all subsequent folders with files belonging to each category
            zip_ref.extractall()"""
