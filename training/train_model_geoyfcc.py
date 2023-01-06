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
import geoyfcc_model
import load_data
from sklearn.metrics import average_precision_score, roc_curve, auc

def auc_final(targets_all, scores_all):
    auc_scores = []
    for i in range(29):
        fpr, tpr, _ = roc_curve(targets_all[:, i], scores_all[:, i])
        roc_aucTest = auc(fpr, tpr)
        auc_scores.append(roc_aucTest)
    return np.mean(auc_scores)

def bootstrap_ap(targets_all, scores_all,  repeat=500):
    max_val = targets_all.squeeze().shape[0]
    avg_prec_weights = np.zeros(repeat)
    avg_prec = np.zeros(repeat)
    for i in range(repeat):
        rand_index = np.random.randint(0, max_val, max_val)
        targets = targets_all[rand_index]
        scores = scores_all[rand_index]
        avg_prec[i] = average_precision_score(targets, scores)
        
    return np.median(avg_prec), np.std(avg_prec)

def bootstrap_acc(targets_all, scores_all,  repeat=500):
    max_val = targets_all.squeeze().shape[0]
    avg_prec_weights = np.zeros(repeat)
    avg_prec = np.zeros(repeat)
    for i in range(repeat):
        rand_index = np.random.randint(0, max_val, max_val)
        targets = targets_all[rand_index]
        scores = scores_all[rand_index]
        avg_prec[i], _ = compute_acc(targets, scores)
        
    return np.median(avg_prec), np.std(avg_prec)

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

geoyfcc_dset = pickle.load(open('data/geoyfcc_prep.pkl', 'rb'))
imagenet_dset = pickle.load(open('data/imagenet29_prep.pkl', 'rb'))


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

geoyfcc_train_names = geoyfcc_dset['train'][0]
geoyfcc_train_obj = geoyfcc_dset['train'][1]
print("GeoYFCC Training Size: " + str(len(geoyfcc_train_names)))

imagenet_train_names = imagenet_dset['train'][0]
imagenet_train_obj = imagenet_dset['train'][1]
print("Imagenet Training Size: " + str(len(imagenet_train_names)))

train_names = [*imagenet_train_names, *geoyfcc_train_names]
train_obj = [*imagenet_train_obj, *geoyfcc_train_obj]
print("Total Training Size: " + str(len(train_names)))

val_names = [*imagenet_dset['val'][0], *geoyfcc_dset['val'][0]]
val_obj = [*imagenet_dset['val'][1], *geoyfcc_dset['val'][1]]
val = [val_names, val_obj]

dset_train = load_data.ImageDatasetMultiLabel(train_names, train_obj, transform = transform_train)
loader_train = load_data.DataLoader(dset_train, **params_train)

dset_val = load_data.ImageDatasetMultiLabel(*val, transform = transform_test)
loader_val = load_data.DataLoader(dset_val, **params_valtest)

criterion = nn.BCEWithLogitsLoss()
ydtype = torch.long

model = geoyfcc_model.ResNet50() 

optimizer_setting = {
    'optimizer': torch.optim.Adam,
    'lr': 0.001
}
optimizer = optimizer_setting['optimizer']( 
                    params=model.parameters(), 
                    lr=optimizer_setting['lr']) 

count=0
if torch.cuda.is_available(): 
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

batch_size = params_train['batch_size']
N = len(train_names)//batch_size + 1

save_dir = 'record/train_on_{}'.format("pretrained_geoyfcc_50epochs")
test_dir = 'record/train_on_{}'.format("pretrained_geoyfcc_50epochs")
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
            loss = criterion(scores, y.float())

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

        acc = auc_final(np.concatenate(all_targets), np.concatenate(all_scores))

        print(e, acc.item(), loss, flush=True)  
        if acc>best_acc:
            best_acc =acc
            torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch': e, 'loss':loss, 'acc':acc}, 
                        '{}/final.pth'.format(save_dir)
                        )

        torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch': e, 'loss':loss, 'acc':best_acc}, 
                    '{}/current.pth'.format(save_dir )
                    )

if args.test:
    print('Testing ... ', flush=True)
    print('Test Set: ' + str(test_dir))
    print("Model Accuracy: " + str(torch.load('{}/final.pth'.format(test_dir))['acc']))
    print("Model Loss: " + str(torch.load('{}/final.pth'.format(test_dir))['loss']))
    model.load_state_dict(torch.load('{}/final.pth'.format(test_dir))['model']) # change
    model.cuda()

    # can change test set to a geode test set as well
    geoyfcc_dset_test = load_data.ImageDatasetMultiLabel(*geoyfcc_dset['test'], transform = transform_test)
    geoyfcc_loader_test = load_data.DataLoader(geoyfcc_dset_test, **params_valtest)

    geoyfcc_all_scores = []
    geoyfcc_all_targets = []
    with torch.no_grad():
        for (x,y) in geoyfcc_loader_test:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=ydtype)
                
            scores = model(x)  
            geoyfcc_all_scores.append(scores.detach().cpu().numpy())
            geoyfcc_all_targets.append(y.detach().cpu().numpy())

    geoyfcc_all_targets = np.concatenate(geoyfcc_all_targets)
    geoyfcc_all_scores = np.concatenate(geoyfcc_all_scores)
    acc, _ = compute_acc(geoyfcc_all_targets,geoyfcc_all_scores)
    print(acc)
