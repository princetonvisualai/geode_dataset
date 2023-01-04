import numpy as np
from sklearn.linear_model import LogisticRegression
import utils
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import argparse
import torch




parser = argparse.ArgumentParser()

parser.add_argument('--reg', type=int, default=0, metavar='reg',
                    help='region number')
parser.add_argument('--perc', type=float, default=0, metavar='reg',
                    help='percent to use for training')
parser.add_argument('--save', type=str, default='test', metavar='reg',
                    help='save directory')
parser.add_argument('--test', action='store_true')

args = parser.parse_args()


imagenet_dset = pickle.load(open('data/imagenet_prep.pkl', 'rb'))
imagenet_indexsample = pickle.load(open('data/imagenet_33383indices.pkl', 'rb'))
imagenet_dset_excluded = pickle.load(open('data/imagenet33383_excludedIndices.pkl', 'rb'))
africa_dset = pickle.load(open('data/appen_prep_africa.pkl', 'rb'))
americas_dset = pickle.load(open('data/appen_prep_americas.pkl', 'rb'))
eastasia_dset = pickle.load(open('data/appen_prep_eastasia.pkl', 'rb'))
southeastasia_dset = pickle.load(open('data/appen_prep_southeastasia.pkl', 'rb'))
westasia_dset = pickle.load(open('data/appen_prep_westasia.pkl', 'rb'))
europe_dset = pickle.load(open('data/appen_prep_europe.pkl', 'rb')) 



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
print("Appen Training Size (29820): " + str(len(region_train_names)))

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

appen = {}

for i in range(6):
    appen[i] = pickle.load(open('data/appen/PASS_features/region{}.pkl'.format(i), 'rb'))

appen_features = {}

for i in range(6):
    for o in appen[i].keys():
        for s in ['train', 'test']:
            appen_features.update(appen[i][o][s])


imagenet_features = {}
import os

for obj in os.listdir('data/im/PASS_features/'):
    print(obj)
    imagenet_features.update(pickle.load(open('data/im/PASS_features/{}'.format(obj), 'rb')))

imagenet_features_2 = {}
for a in imagenet_features:
    #print(a, imagenet_features[a].shape)
    #break
    imagenet_features_2[a.split('/')[-1]] = imagenet_features[a]
    
features_dict = appen_features 
features_dict.update(imagenet_features_2)

train_features = []

for a in train_names:
    if a in features_dict:
        train_features.append(features_dict[a])
    elif a.split('/')[-1] in features_dict:
        train_features.append(features_dict[a.split('/')[-1]])
    else:
        print('missing :{}'.format(a))
train_features = torch.Tensor(np.concatenate(train_features)).squeeze()
print(train_features.shape)
val_features = []

for a in val_names:
    if a in features_dict:
        val_features.append(features_dict[a])
    elif a.split('/')[-1] in features_dict:
        val_features.append(features_dict[a.split('/')[-1]])
    else:
        print('missing :{}'.format(a))
 
val_features = torch.Tensor(np.concatenate(val_features)).squeeze()
print(val_features.shape)



criterion = torch.nn.CrossEntropyLoss()
#acc_function = compute_acc 
ydtype = torch.long

m = torch.nn.Linear(2048, 38)  

optimizer_setting = {
    'optimizer': torch.optim.SGD,
    'lr': 0.1,
    'momentum': 0.9
}
optimizer = optimizer_setting['optimizer']( 
                    params=m.parameters(), 
                    lr=optimizer_setting['lr'], momentum=optimizer_setting['momentum']) 
count=0
if torch.cuda.is_available(): 
    device = torch.device('cuda')
    m = m.to(device)
else:
    device = torch.device('cpu')

if not os.path.exists(args.save):
    os.makedirs(args.save)

train_obj = torch.Tensor(train_obj)
val_obj = torch.Tensor(val_obj)

dtype = torch.float32
best_acc = 0.0


if not args.test:
    

    batch_size = 512
    N = len(train_features)//batch_size+1
    for e in range(500):
        m.train()
        rand_perm = np.random.permutation(len(train_features))

        train_features = train_features[rand_perm]
        train_obj = train_obj[rand_perm]
        train_names = [train_names[i] for i in rand_perm]
        
        for t in range(N):
            feat_batch = train_features[t*batch_size:(t+1)*batch_size].to(device=device, dtype = dtype) 
            target_batch = train_obj[t*batch_size:(t+1)*batch_size].to(device=device, dtype = ydtype) 
            
            sc = m(feat_batch)

            optimizer.zero_grad()
            
            loss = criterion(sc, target_batch)
            loss.backward()

            optimizer.step()

            #if t%100==0:
            #    print(e, t, loss, flush=True)

        
        m.eval()

        val_scores = m(val_features.to(device))
        loss_val = criterion(val_scores, val_obj.to(device, ydtype))

        val_pred = np.argmax(val_scores.squeeze().detach().cpu().numpy(), axis=1)
        acc = np.where(val_pred==val_obj.numpy(), 1, 0).mean() 
        
        print(e, loss, acc, flush=True)

        if acc>best_acc:
            best_acc =acc
            torch.save({'model':m.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch': e, 'loss':loss, 'acc':acc}, 
                        '{}/final.pth'.format(args.save)
                        )

        torch.save({'model':m.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch': e, 'loss':loss, 'acc':best_acc}, 
                    '{}/current.pth'.format(args.save)
                    )


region_to_name = {0:'africa', 1:'americas', 2:'eastasia', 3:'europe', 4:'southeastasia', 5:'westasia'}      
m.load_state_dict(torch.load('{}/final.pth'.format(args.save))['model'])
m = m.to(device)
"""


imgnet_val_feat = []

for a in imagenet_dset['test'][0]:
    if a in features_dict:
        imgnet_val_feat.append(features_dict[a])
    elif a.split('/')[-1] in features_dict:
        imgnet_val_feat.append(features_dict[a.split('/')[-1]])
    else:
        print('missing :{}'.format(a))
 
imgnet_val_feat = torch.Tensor(np.concatenate(imgnet_val_feat)).squeeze().to(device)
imgnet_val_target = np.array(imagenet_dset['test'][1])

print(imgnet_val_target.shape)
imgnet_val_sc = m(imgnet_val_feat)
imgnet_val_pred = np.argmax(imgnet_val_sc.detach().cpu().numpy(), axis=1)

print(imgnet_val_pred.shape)
print('imgnet_acc', np.mean(np.where(imgnet_val_pred.squeeze() == np.array(imgnet_val_target), 1, 0)))
"""

"""
ds_pickle = pickle.load(open('dollarstreet_pass_feat.pkl', 'rb'))
mapping_object = pickle.load(open('ds_to_appen.pkl', 'rb'))
mapping_region = pickle.load(open('ds_to_region.pkl', 'rb'))


score_region = {i:{} for i in range(10)}

for a in ds_pickle:
    feat = torch.Tensor(ds_pickle[a]['score']).reshape(-1, 2048).to(device)
    
    reg = mapping_region[ds_pickle[a]['country']]
    target = mapping_object[ds_pickle[a]['target']]
    
    if target not in score_region[reg]:
        score_region[reg][target] = {'correct':0, 'total':0}

    mod_sc = m(feat).detach().cpu().numpy()
    pred = np.argmax(mod_sc.squeeze())

    score_region[reg][target]['total']+=1
    if pred == target:
        score_region[reg][target]['correct']+=1


rev_mapping_object = {v:k for (k, v) in mapping_object.items()}

for i in range(10):
    for key in score_region[i]:
        print(i, rev_mapping_object[key], score_region[i][key]['correct'], score_region[i][key]['total'], sep=',\t')

"""


for i in range(6):
    region_dset = pickle.load(open('data/appen_prep_{}.pkl'.format(region_to_name[i]), 'rb')) # change for region
    
    region = pickle.load(open('data/appen/PASS_features/region{}.pkl'.format(i), 'rb'))

    reg_features_dict = {}

    for o in region.keys():
        for s in ['train', 'test']:
            reg_features_dict.update(region[o][s])


    softmax = torch.nn.Softmax(dim=1)
    test_region = []

    for a in region_dset['val'][0]:
        test_region.append(reg_features_dict[a])

    test_region = torch.Tensor(np.concatenate(test_region)).squeeze()
    print(test_region.shape)
    m.eval()
    sc_region = softmax(m(test_region.to(device))).detach().cpu().numpy()
    print(sc_region.shape, len(region_dset['test'][1]))

    pred_region = np.argmax(sc_region, axis=1).squeeze()

    acc_region = np.where(pred_region==np.array(region_dset['test'][1]), 1, 0).mean()

    with open('{}/region{}_test_scores.pkl'.format(args.save, i), 'wb+') as handle:
        pickle.dump(sc_region, handle)
    print('Region {} test scores: '.format(i), 100*acc_region)
        

