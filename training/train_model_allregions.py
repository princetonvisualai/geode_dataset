import torch
import torch.nn as nn
import pickle
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import torchvision.transforms as T
import utils_local
import model
import load_data
from sklearn.metrics import average_precision_score

def compute_acc(target, predicted, weighted=False):
    pred_idx = np.argmax(predicted, axis=1)
    accuracy = np.mean(target==pred_idx)

    return accuracy, '__'

def compute_AP(target, predicted, weighted=False):
    per_class_AP = []
    for i in range(target.shape[1]):
        if not target[:, i].sum()==0: 
            weights = None
            if weighted:   
                weights = target[:, i]*(target[:, i].shape[0]/(2*target[:,i].sum())) + (  
                            1-target[:,i])*(target[:, i].shape[0]/(2*(target[:, i].shape[0]-target[:, i].sum())))
            per_class_AP.append(average_precision_score(target[:, i], predicted[:, i], sample_weight=weights))

    return np.mean(per_class_AP), per_class_AP

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()


params_train = {'batch_size': 32,
         'shuffle': True,
         'num_workers': 0}

params_valtest = {'batch_size': 32,
         'shuffle': False,
         'num_workers': 0}

imagenet_dset = pickle.load(open('data/imagenet_prep.pkl', 'rb'))
imagenet_indexsample = pickle.load(open('data/imagenet_33383indices.pkl', 'rb'))
imagenet_dset_excluded = pickle.load(open('data/imagenet33383_excludedIndices.pkl', 'rb'))
africa_dset = pickle.load(open('data/geode_prep_africa.pkl', 'rb'))
americas_dset = pickle.load(open('data/geode_prep_americas.pkl', 'rb'))
eastasia_dset = pickle.load(open('data/geode_prep_eastasia.pkl', 'rb'))
southeastasia_dset = pickle.load(open('data/geode_prep_southeastasia.pkl', 'rb'))
westasia_dset = pickle.load(open('data/geode_prep_westasia.pkl', 'rb'))
europe_dset = pickle.load(open('data/geode_prep_europe.pkl', 'rb'))



normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

transform_train = T.Compose([
    T.Resize(256),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    normalize
])

transform_test = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    normalize
])

print('Building Training Sets ... ', flush=True)

region_num_images = 4970
africa_random_sample = np.random.choice(len(africa_dset['train'][0]), region_num_images, replace=False)
americas_random_sample = np.random.choice(len(americas_dset['train'][0]), region_num_images, replace=False)
eastasia_random_sample = np.random.choice(len(eastasia_dset['train'][0]), region_num_images, replace=False)
southeastasia_random_sample = np.random.choice(len(southeastasia_dset['train'][0]), region_num_images, replace=False)
westasia_random_sample = np.random.choice(len(westasia_dset['train'][0]), region_num_images, replace=False)
europe_random_sample = np.random.choice(len(europe_dset['train'][0]), region_num_images, replace=False)

imagenet_num_images_ratio = 8533/len(imagenet_indexsample['index'][0])
imagenet_random_sample = []

full_38_cat = []
for obj in range(38):
    category_pics = []
    for pic_index in imagenet_indexsample['index'][0]:
        if imagenet_dset['train'][1][pic_index] == obj:
            category_pics.append(pic_index)
    full_38_cat.append(category_pics)
    
total = 0
for lst in full_38_cat:
    total = total + len(lst)
print("Checkpoint1 (33383): " + str(total))

excluded = []
for lst in full_38_cat:
    num_in_category = len(lst)
    num_images = np.floor(imagenet_num_images_ratio*num_in_category).astype(int)
    subset_sample = np.random.choice(lst, num_images, replace=False)
    excluded_subset = [i for i in lst if i not in subset_sample]
    imagenet_random_sample = [*imagenet_random_sample, *subset_sample]
    for excluded_idx in excluded_subset:
        excluded.append(excluded_idx)
    
print("Number of Excluded Images Checkpoint1: " + str(len(excluded)))
print("Number of Excluded Images Checkpoint2: " + str(len(imagenet_indexsample['index'][0]) - len(imagenet_random_sample)))

needed = 8533 - len(imagenet_random_sample)
addOn_sample = np.random.choice(excluded, needed, replace=False)
imagenet_random_sample = [*imagenet_random_sample, *addOn_sample]
imagenet_random_sample = np.unique(np.array(imagenet_random_sample))

africa_train_names = [africa_dset['train'][0][i] for i in africa_random_sample]
africa_train_obj = [africa_dset['train'][1][i] for i in africa_random_sample]

americas_train_names = [americas_dset['train'][0][i] for i in americas_random_sample]
americas_train_obj = [americas_dset['train'][1][i] for i in americas_random_sample]

eastasia_train_names = [eastasia_dset['train'][0][i] for i in eastasia_random_sample]
eastasia_train_obj = [eastasia_dset['train'][1][i] for i in eastasia_random_sample]

southeastasia_train_names = [southeastasia_dset['train'][0][i] for i in southeastasia_random_sample]
southeastasia_train_obj = [southeastasia_dset['train'][1][i] for i in southeastasia_random_sample]

westasia_train_names = [westasia_dset['train'][0][i] for i in westasia_random_sample]
westasia_train_obj = [westasia_dset['train'][1][i] for i in westasia_random_sample]

europe_train_names = [europe_dset['train'][0][i] for i in europe_random_sample]
europe_train_obj = [europe_dset['train'][1][i] for i in europe_random_sample]

region_train_names = [*africa_train_names, *americas_train_names, *eastasia_train_names, *southeastasia_train_names, *westasia_train_names, *europe_train_names]
region_train_obj = [*africa_train_obj, *americas_train_obj, *eastasia_train_obj, *southeastasia_train_obj, *westasia_train_obj, *europe_train_obj]
print("GeoDE Training Size (29820): " + str(len(region_train_names)))

imagenet_train_names = [imagenet_dset['train'][0][i] for i in imagenet_random_sample]
imagenet_train_obj = [imagenet_dset['train'][1][i] for i in imagenet_random_sample]
print("Imagenet Training Size (8533): " + str(len(imagenet_train_names)))

train_names = [*imagenet_train_names, *region_train_names]
train_obj = [*imagenet_train_obj, *region_train_obj]
print("Total Training Size (38353): " + str(len(train_names)))

region_val_names = [*africa_dset['val'][0], *americas_dset['val'][0], *eastasia_dset['val'][0], *southeastasia_dset['val'][0], *westasia_dset['val'][0], *europe_dset['val'][0]]
region_val_obj = [*africa_dset['val'][1], *americas_dset['val'][1], *eastasia_dset['val'][1], *southeastasia_dset['val'][1], *westasia_dset['val'][1], *europe_dset['val'][1]]

val_names = [*imagenet_dset['val'][0], *region_val_names]
val_obj = [*imagenet_dset['val'][1], *region_val_obj]
val = [val_names, val_obj]

dset_train = load_data.ImageDataset_imagenet(train_names, train_obj, transform = transform_train)
loader_train = load_data.DataLoader(dset_train, **params_train)

dset_val = load_data.ImageDataset_imagenet(*val, transform = transform_test)
loader_val = load_data.DataLoader(dset_val, **params_valtest)

criterion = nn.CrossEntropyLoss()
ydtype = torch.long

model = model.ResNet50() 

optimizer_setting = {
    'optimizer': torch.optim.SGD,
    'lr': 0.1,
    'momentum': 0.9
}
optimizer = optimizer_setting['optimizer']( 
                    params=model.parameters(), 
                    lr=optimizer_setting['lr'], momentum=optimizer_setting['momentum']) 
count=0
if torch.cuda.is_available(): 
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

batch_size = params_train['batch_size']
N = len(train_names)//batch_size + 1

save_dir = 'record/train_on_{}'.format("pretrained_imagenet_allregions_50epochs")
test_dir = 'record/test_on_{}'.format("pretrained_imagenet_allregions_50epochs")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

dtype = torch.float32
best_acc = 0.0

if not args.test:
    epoch_beg = 0

    """
    if path.exists('{}/current.pth'.format(save_dir)):
        A = torch.load('{}/current.pth'.format(save_dir))
        model.load_state_dict(A['model'])
        optimizer.load_state_dict(A['optimizer'])
        epoch_beg = A['epoch']
    """

    print('Training ... ', flush=True)
    if device==torch.device('cuda'):
        model.cuda()
    for e in range(epoch_beg, 50):
        model.train()

        for (x, y) in loader_train:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=ydtype)

            scores = model(x)  
            loss = criterion(scores, y) 
           
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        
        all_scores = []
        all_targets = []
         
        with torch.no_grad():
            for (x,y) in loader_val:
                x = x.to(device=device, dtype=dtype)
                y = y.to(device=device, dtype=ydtype)
                
                scores = model(x)  
                all_scores.append(scores.detach().cpu().numpy())
                all_targets.append(y.detach().cpu().numpy())

        acc, _ = compute_acc(np.concatenate(all_targets), np.concatenate(all_scores))

        print(e, acc.item(), loss, flush=True)  
        if acc>best_acc:
            best_acc =acc
            torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch': e, 'loss':loss, 'acc':acc}, 
                        '{}/final.pth'.format(save_dir)
                        )

        torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch': e, 'loss':loss, 'acc':best_acc}, 
                    '{}/current.pth'.format(save_dir )
                    )
