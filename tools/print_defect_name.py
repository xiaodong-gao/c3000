import os

folder = r"./images"

for name in os.listdir(folder):
    path = os.path.join(folder, name)
    if os.path.isfile(path):
        print(name)