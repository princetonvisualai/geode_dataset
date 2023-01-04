import numpy as np
from sklearn.linear_model import LogisticRegression
import utils
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import argparse
import torch


def get_data(region_dset, imagenet_dset, imagenet_indexsample, imagenet_dset_excluded, ratio):
    print('Building Training Sets ... ', flush=True)

    region_num_images = 4970

    print("Percentage of Appen Images: " + str(ratio))

    full_region_random_sample = np.random.choice(len(region_dset['train'][0]), region_num_images, replace=False) #4970 images

    region_num_images = np.floor(region_num_images * ratio).astype(int) #0.5
    region_random_sample = np.random.choice(full_region_random_sample, region_num_images, replace=False) # gives indices corr to region_dset

    excluded_pics = np.array([i for i in full_region_random_sample if i not in region_random_sample]) # appen excluded from training from org 4970

    checkpoint1 = len(full_region_random_sample) - len(region_random_sample)

    print("Number of Images Taken out of Appen Checkpoint 1: " + str(checkpoint1))
    print("Number of Images Taken out of Appen Checkpoint 2: " + str(len(excluded_pics)))

    imagenet_addOn = set() # imagenet images added on to supplement appen being taken out
    imagenet_excluded = set() # images excluded from 33383
    for obj in range(38):
        for image in imagenet_dset_excluded[str(obj)][0]:
            imagenet_excluded.add(image)
        excluded = set() 
        for pic_index in excluded_pics: # for each index in pics excluded from appen 4970
            excluded_obj = region_dset['train'][1][pic_index]
            if excluded_obj == obj:
                excluded.add(pic_index)
        if len(imagenet_dset_excluded[str(obj)][0]) < len(excluded):
            addOn_sample = np.random.choice(imagenet_dset_excluded[str(obj)][0], len(imagenet_dset_excluded[str(obj)][0]), replace=False)
        else:
            addOn_sample = np.random.choice(imagenet_dset_excluded[str(obj)][0], len(excluded), replace=False)
        for image in addOn_sample:
            imagenet_addOn.add(image)
        
    available_imagenet = np.array([i for i in imagenet_excluded if i not in imagenet_addOn])
    addon_subset = np.random.choice(available_imagenet, (len(excluded_pics) - len(imagenet_addOn)), replace=False)
    for image in addon_subset:
        imagenet_addOn.add(image)

    print("Number of Images Taken out of Appen Checkpoint 3: " + str(len(imagenet_addOn)))

    imagenet_random_sample = [*imagenet_indexsample['index'][0], *imagenet_addOn]

    region_train_names = [region_dset['train'][0][i] for i in region_random_sample]
    region_train_obj = [region_dset['train'][1][i] for i in region_random_sample]
    print("Appen Training Size: " + str(len(region_train_names)))

    imagenet_train_names = [imagenet_dset['train'][0][i] for i in imagenet_random_sample]
    imagenet_train_obj = [imagenet_dset['train'][1][i] for i in imagenet_random_sample]
    print("Imagenet Training Size: " + str(len(imagenet_train_names)))

    train_names = [*imagenet_train_names, *region_train_names]
    train_obj = [*imagenet_train_obj, *region_train_obj]
    print("Total Training Size: " + str(len(train_names)))

    val_names = [*imagenet_dset['val'][0], *region_dset['val'][0]]
    val_obj = [*imagenet_dset['val'][1], *region_dset['val'][1]]
    val = [val_names, val_obj]

    return train_names, train_obj, val_names, val_obj





parser = argparse.ArgumentParser()

parser.add_argument('--reg', type=int, default=0, metavar='reg',
                    help='region number')
parser.add_argument('--perc', type=float, default=0, metavar='reg',
                    help='percent to use for training')
parser.add_argument('--save', type=str, default='test', metavar='reg',
                    help='save directory')
parser.add_argument('--test', action='store_true')

args = parser.parse_args()




params_train = {'batch_size': 32,
         'shuffle': True,
         'num_workers': 0}

params_valtest = {'batch_size': 32,
         'shuffle': False,
         'num_workers': 0}

region_to_name = {0:'africa', 1:'americas', 2:'eastasia', 3:'europe', 4:'southeastasia', 5:'westasia'}


print('Region:{}, perc: {}'.format(region_to_name[args.reg], args.perc))

region_dset = pickle.load(open('data/appen_prep_{}.pkl'.format(region_to_name[args.reg]), 'rb')) # change for region
imagenet_dset = pickle.load(open('data/imagenet_prep.pkl', 'rb'))
imagenet_indexsample = pickle.load(open('data/imagenet_33383indices.pkl', 'rb'))
imagenet_dset_excluded = pickle.load(open('data/imagenet33383_excludedIndices.pkl', 'rb'))


train_names, train_obj, val_names, val_obj = get_data(region_dset, 
                                                      imagenet_dset, 
                                                      imagenet_indexsample, 
                                                      imagenet_dset_excluded, 
                                                      args.perc)

appen = pickle.load(open('data/appen/PASS_features/region{}.pkl'.format(args.reg), 'rb'))

appen_features = {}

for o in appen.keys():
    for s in ['train', 'test']:
        appen_features.update(appen[o][s])


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

batch_size = params_train['batch_size']
N = len(train_names)//batch_size + 1

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
            

m.load_state_dict(torch.load('{}/final.pth'.format(args.save))['model'])
m = m.to(device)
europe_dset = pickle.load(open('data/appen_prep_{}.pkl'.format('europe'), 'rb')) # change for region

europe = pickle.load(open('data/appen/PASS_features/region{}.pkl'.format(3), 'rb'))

eur_features_dict = {}

for o in europe.keys():
    for s in ['train', 'test']:
        eur_features_dict.update(europe[o][s])

val_europe = []

for a in europe_dset['val'][0]:
    val_europe.append(eur_features_dict[a])

val_europe = torch.Tensor(np.concatenate(val_europe)).squeeze()
m.eval()
softmax = torch.nn.Softmax(dim=1)
sc_europe = softmax(m(val_europe.to(device))).detach().cpu().numpy()

pred_europe = np.argmax(sc_europe, axis=1).squeeze()

acc_europe = np.where(pred_europe==np.array(europe_dset['val'][1]), 1, 0).mean()

with open('{}/europe_val_scores.pkl'.format(args.save), 'wb+') as handle:
    pickle.dump(sc_europe, handle)
print('Europe val scores: ', 100*acc_europe)


val_region = []

for a in region_dset['val'][0]:
    val_region.append(appen_features[a])

val_region = torch.Tensor(np.concatenate(val_region)).squeeze()
m.eval()
sc_region = softmax(m(val_region.to(device))).detach().cpu().numpy()

pred_region = np.argmax(sc_region, axis=1).squeeze()

acc_region = np.where(pred_region==np.array(region_dset['val'][1]), 1, 0).mean()

with open('{}/region{}_val_scores.pkl'.format(args.save, args.reg), 'wb+') as handle:
    pickle.dump(sc_region, handle)
print('Region val scores: ', 100*acc_region)
    


test_europe = []

for a in europe_dset['test'][0]:
    test_europe.append(eur_features_dict[a])

test_europe = torch.Tensor(np.concatenate(test_europe)).squeeze()
m.eval()
softmax = torch.nn.Softmax(dim=1)
sc_europe = softmax(m(test_europe.to(device))).detach().cpu().numpy()

pred_europe = np.argmax(sc_europe, axis=1).squeeze()

acc_europe = np.where(pred_europe==np.array(europe_dset['test'][1]), 1, 0).mean()

with open('{}/europe_test_scores.pkl'.format(args.save), 'wb+') as handle:
    pickle.dump(sc_europe, handle)
print('Europe test scores: ', 100*acc_europe)


test_region = []

for a in region_dset['test'][0]:
    test_region.append(appen_features[a])

test_region = torch.Tensor(np.concatenate(test_region)).squeeze()
m.eval()
sc_region = softmax(m(test_region.to(device))).detach().cpu().numpy()

pred_region = np.argmax(sc_region, axis=1).squeeze()

acc_region = np.where(pred_region==np.array(region_dset['test'][1]), 1, 0).mean()

with open('{}/region{}_test_scores.pkl'.format(args.save, args.reg), 'wb+') as handle:
    pickle.dump(sc_region, handle)
print('Region test scores: ', 100*acc_region)
    

