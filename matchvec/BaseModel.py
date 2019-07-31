import boto3
import os.path as osp
import os
bucket_name = 'iaflash'

class BaseModel():

    def download_model_folder(self, dst_path, src_path):

        if not osp.exists(dst_path):
            os.makedirs(dst_path)

        for file in self.files:
            if not osp.isfile(osp.join(dst_path, file)):
                print("Downloading %s"%osp.join(src_path, file))
                s3 = boto3.resource('s3')
                myobject = s3.Object(bucket_name, osp.join(src_path, file))
                myobject.download_file(osp.join(dst_path, file))
                print("Downloading ok\n")
            else:
                print("Model already locally")
