import os
import csv

# collect directory name and store in dirs(list)
path = "/media/mmlab206/YT8M-4TB/aicup_data/aicup/"
dirs = sorted(os.listdir(path))

# decide how to split train and valid dataset
split_ratio = 0.8

# create categorys.csv,train.csv,valid.csv
with open('categories.csv','w') as cat_csv,open('train.csv','w') as train_csv,open('valid.csv','w')as valid_csv:
   
   # go through all images under each folder
   for category_id, dir_name in enumerate(dirs):

      dir_path = os.path.join(path, dir_name)
      
      # list for image name
      images = os.listdir(dir_path)[:1000]
      
      # train dataset size
      train_size = int(len(images)*split_ratio)
      
      write = csv.writer(cat_csv)
      write.writerow([dir_name])
      
      for i, image_name in enumerate(images):
         
         if i > train_size:
            write = csv.writer(valid_csv)
            write.writerow([image_name, category_id])

         else:
            write = csv.writer(train_csv)
            write.writerow([image_name, category_id])
         
