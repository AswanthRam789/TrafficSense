import os

dataset = r"dataset/archive (1)/data"

print("Classes Found:\n")

for folder in os.listdir(dataset):
    path = os.path.join(dataset, folder)

    if os.path.isdir(path):
        print(folder, ":", len(os.listdir(path)), "images")