import shutil
import kagglehub
import os
import shutil

# Download latest version
ori_path = kagglehub.dataset_download("blastchar/telco-customer-churn")

print("Path to dataset files:", ori_path)

# Get the first file in the directory
ori_file = os.listdir(ori_path)[0]
path_ori_file = os.path.join(ori_path, ori_file)

# Moved the file to customer_churn_raw directory and renamed it
dst_path = "./customer_churn_raw"
new_file_name = "customer_churn.csv"

path_dst_file = os.path.join(dst_path, new_file_name)

shutil.move(path_ori_file, path_dst_file)

print(f"File {ori_file} has been moved to {dst_path}")