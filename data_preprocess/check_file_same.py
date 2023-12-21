import os
import shutil

def is_image_file(filename):
    return any(filename.endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])

path1 = './images'
path2 = './xml'
out = './error_data'
files1_original = []
files2_original = []
os.makedirs(out, exist_ok=True)

for names in os.listdir(path1):
    if is_image_file(names):
        file_base_name = os.path.splitext(names)[0]
        files1_original.append(file_base_name)
        print(file_base_name)

for names in os.listdir(path2):
    if names.endswith('.xml'):
        files2_original.append(os.path.splitext(names)[0])
        print(os.path.splitext(names)[0])

out1 = []
out2 = []

for names1 in files1_original:
    if names1 not in files2_original:
        out1.append(names1)
        with open(os.path.join(out, 'out.txt'), 'a') as fd:
            fd.write(names1 + '\n')

for names2 in files2_original:
    if names2 not in files1_original:
        out2.append(names2)
        with open(os.path.join(out, 'out.txt'), 'a') as fd:
            fd.write(names2 + '\n')

for i in out1:
    for ext in ['.jpg', '.jpeg', '.png']:
        src_path = os.path.join(path1, i + ext)
        if os.path.exists(src_path):
            shutil.move(src_path, out)
            break

for i in out2:
    shutil.move(os.path.join(path2, i + '.xml'), out)