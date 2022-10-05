from dataset import AICUPdataset
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T

with open("/home/mmlab206/aicup/categories.csv", 'r') as f:
	categories = f.readlines()
	categories = [i.strip('\n') for i in categories]

# load valid.csv 產生aicupdataset
valid_csv = "valid.csv"
image_dir = "/media/mmlab206/YT8M-4TB/aicup_data/aicup"
aicup_dataset = AICUPdataset(
	valid_csv, 
	image_dir,
	transform= T.Compose([
        T.Resize([160, 160]),
        T.ToTensor(),
    ]),
	)
batch = 16
aicup_dataloader = DataLoader(
	aicup_dataset,
	batch_size=batch,
	)
# create image name list
images_name = aicup_dataset.annotations.iloc[:,0]
images_name = [i for i in images_name]

# load weight變成可用model
weight_file = "/home/mmlab206/aicup/output/resnet18_160/best_model.pth"
parameters = torch.load(weight_file)
model = torchvision.models.resnet18(num_classes=33)
model.load_state_dict(parameters)
model.eval()
model.to("cuda")

# 將要使用之結果寫進csv
with open("result.csv", "w") as f:
    for index, (images, labels) in enumerate(aicup_dataloader):
        output = model(images.to("cuda"))

        # category id
        _, ground_truth = torch.max(labels, 1)
        _, pred = torch.max(output, 1)
        # image_name = images_name[index:index+8]
        try:
            images = images_name[index*batch:(index+1)*batch]
        except:
            images = images_name[index*batch:]

        for image_name, gt, pd in zip(images, ground_truth,pred):

            if gt.item() == pd.item():
                correctness = 1
            else:
                correctness = 0

            # print(image_name, gt.item(), pd.item())
            print(image_name, categories[gt.item()], categories[pd.item()])
        
            f.write(f"{image_name},{categories[gt.item()]},{categories[pd.item()]}, {correctness}\n")

# image_name, ground_truth, predict
