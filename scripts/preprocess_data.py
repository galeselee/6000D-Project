import os

root_dir = "./test_data"
new_dir = "collected_data"
# root_dir = argv[1]
# new_dir = argv[2]
os.system(f"mkdir {new_dir}")

for file in os.listdir(root_dir):
    d = os.path.join(root_dir, file)
    for mat_file in os.listdir(d):
        matrix = os.path.join(d, mat_file)
        os.system(f"cp {matrix} ./{new_dir}")
