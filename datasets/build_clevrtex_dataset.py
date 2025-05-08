import json
import h5py
from PIL import Image
from skimage.transform import resize
import numpy as np
import sys
import os


#enumerate name of blend files in materials dir
materials_dir = './materials/'
materials = os.listdir(materials_dir)
#remove .blend extension and convert to lower case
materials = [m[:-6].lower() for m in materials if m.endswith('.blend')]
materials_exclude = ['mymetal','rubber']
materials = [m for m in materials if m not in materials_exclude]

#build program space

program_space = {
    "shapes":["cube", "cylinder", "sphere", "monkey"],
    "sizes":["small", "medium", "large"],
    "materials":materials,
}
label2program = {
    "sizes": {i:program_space["sizes"][i] for i in range(len(program_space["sizes"]))},
    "shapes": {i:program_space["shapes"][i] for i in range(len(program_space["shapes"]))},
    "materials": {i:program_space["materials"][i] for i in range(len(program_space["materials"]))}
}

program2label = {
    "sizes": {program_space["sizes"][i]:i for i in range(len(program_space["sizes"]))},
    "shapes": {program_space["shapes"][i]:i for i in range(len(program_space["shapes"]))},
    "materials": {program_space["materials"][i]:i for i in range(len(program_space["materials"]))}
}

#save program space
with open('./program_space.json', 'w') as f:
    json.dump(program_space, f)

with open('./program2label.json', 'w') as f:
    json.dump(program2label, f)

with open('./label2program.json', 'w') as f:
    json.dump(label2program, f)

#save programs
labels = []
img_template = '%%0%dd' %(6)
count = 0
for count in range(50000):
    batch = count//1000
    dir_name = f'./clevrtexv2_full/{batch}/CLEVRTEXv2_full_{img_template % count}/CLEVRTEXv2_full_{img_template % count}.json'
    scene = json.load(open(dir_name))
    num_objects = scene["num_objects"]
    label_list = []
    for object in scene['objects']:
        label = [program2label['sizes'][object['size']], program2label['shapes'][object['shape']], program2label['materials'][object['material']]]
        label += object['3d_coords']
        label_list.append(label)
    labels.append(label_list)

np.save('./train_labels.npy',np.array(labels[:37500])) 
np.save('./val_labels.npy',np.array(labels[37500:40000])) 
np.save('./test_labels.npy',np.array(labels[40000:])) 

#save images

count = 0
images = []
for count in range(50000):
    batch = count//1000
    dir_name = f'./clevrtexv2_full/{batch}/CLEVRTEXv2_full_{img_template % count}/CLEVRTEXv2_full_{img_template % count}.json'
    scene = json.load(open(dir_name))
    num_objects = scene["num_objects"]
    img_path = f'./clevrtexv2_full/{batch}/CLEVRTEXv2_full_{img_template % count}/CLEVRTEXv2_full_{img_template % count}_0003.png'
    img = Image.open(img_path)
    img = np.array(img)
    image = resize(img, (128, 128))
    images.append(image[:,:,:3])

with h5py.File('./images.h5', 'w') as f:
    f.create_dataset('train', data=np.stack(images[:37500],0))
    f.create_dataset('val', data=np.stack(images[37500:40000],0))
    f.create_dataset('test', data=np.stack(images[40000:],0))