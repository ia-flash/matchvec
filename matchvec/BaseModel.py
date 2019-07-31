import boto3
import os.path as osp

bucket_name = 'iaflash'

class BaseModel():

    def download_model_folder(self, dst_path, src_path):

        for file in self.files:
            print("Downloading %s"%file)
            s3 = boto3.resource('s3')
            myobject = s3.Object(bucket_name, osp.join(src_path, file))
            myobject.download_file(osp.join(src_path, file))
            print("Downloading ok\n")
