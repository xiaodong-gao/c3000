import os

folder = r"../images"
txt_path = r"../image_names.txt"

with open(txt_path, "w", encoding="utf-8") as f:
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            print(name)
            f.write(name + "\n")