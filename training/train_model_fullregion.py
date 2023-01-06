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

region_dset = pickle.load(open('data/geode_prep_africa.pkl', 'rb')) # change per region
imagenet_dset = pickle.load(open('data/imagenet_prep.pkl', 'rb'))
imagenet_indexsample = pickle.load(open('data/imagenet_33383indices.pkl', 'rb'))


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
region_random_sample = np.random.choice(len(region_dset['train'][0]), region_num_images, replace=False)

imagenet_random_sample = np.unique(np.array(imagenet_indexsample['index'][0]))

region_train_names = [region_dset['train'][0][i] for i in region_random_sample]
region_train_obj = [region_dset['train'][1][i] for i in region_random_sample]
print("GeoDE Training Size: " + str(len(region_train_names)))

imagenet_train_names = [imagenet_dset['train'][0][i] for i in imagenet_random_sample]
imagenet_train_obj = [imagenet_dset['train'][1][i] for i in imagenet_random_sample]
print("Imagenet Training Size: " + str(len(imagenet_train_names)))

train_names = [*imagenet_train_names, *region_train_names]
train_obj = [*imagenet_train_obj, *region_train_obj]
print("Total Training Size: " + str(len(train_names)))

val_names = [*imagenet_dset['val'][0], *region_dset['val'][0]]
val_obj = [*imagenet_dset['val'][1], *region_dset['val'][1]]
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

save_dir = 'record/train_on_{}'.format("pretrained_imagenet33383_africa38_50epochs") #change per region
test_dir = 'record/train_on_{}'.format("pretrained_imagenet33383_africa38_50epochs") #change per region 
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
            print(loss)

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

if args.test:
    print('Testing ... ', flush=True)
    print('Test Set: ' + str(test_dir))
    print("Model Accuracy: " + str(torch.load('{}/final.pth'.format(test_dir))['acc']))
    model.load_state_dict(torch.load('{}/final.pth'.format(test_dir))['model'])
    model.cuda()

    europe_dset = pickle.load(open('data/geode_prep_europe.pkl', 'rb'))

    europe_dset_test = load_data.ImageDataset_imagenet(*europe_dset['test'], transform = transform_test)
    europe_loader_test = load_data.DataLoader(europe_dset_test, **params_valtest)

    region_dset_test = load_data.ImageDataset_imagenet(*region_dset['test'], transform = transform_test)
    region_loader_test = load_data.DataLoader(region_dset_test, **params_valtest)

    #region
    region_all_scores = []
    region_all_targets = []
    with torch.no_grad():
        for (x,y) in region_loader_test:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=ydtype)

            scores = model(x)
            region_all_scores.append(scores.detach().cpu().numpy())
            region_all_targets.append(y.detach().cpu().numpy())

    region_all_targets = np.concatenate(region_all_targets)
    region_all_scores = np.concatenate(region_all_scores)

    print(region_all_scores.shape)
    print(region_all_scores)
    print('XXXXXX')
    print("Precisions")
    print()

    median = []
    upper_conf = []
    lower_conf = []
    stdev = []
    for obj in range(38):
        per_reg_obj_i = np.where(region_all_targets==obj, 1, 0)
        med, std = bootstrap_ap(per_reg_obj_i, region_all_scores[:, obj])
        median.append(med * 100)
        stdev.append(std)
        upper = (med.item() + (2 * std.item())) * 100
        lower = (med.item() - (2 * std.item())) * 100
        upper_conf.append(upper)
        lower_conf.append(lower)
    
    print("median")
    for num in median:
        print(num)
    print()
    print("upper_conf")
    for num in upper_conf:
        print(num)
    print()
    print("lower_conf")
    for num in lower_conf:
        print(num)
    print()
    print("std without scaling by 100")
    for std in stdev:
        print(std)
    print()

    print('XXXXXX')
    with open('{}/region_test_scores.pkl'.format(test_dir), 'wb+') as handle: # change per region
        pickle.dump({'scores':region_all_scores, 'targets':region_all_targets}, handle)

    #europe
    europe_all_scores = []
    europe_all_targets = []
    with torch.no_grad():
        for (x,y) in europe_loader_test:
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=ydtype)

            scores = model(x)
            europe_all_scores.append(scores.detach().cpu().numpy())
            europe_all_targets.append(y.detach().cpu().numpy())

    europe_all_targets = np.concatenate(europe_all_targets)
    europe_all_scores = np.concatenate(europe_all_scores)

    print(europe_all_scores.shape)
    print('XXXXXX')

    europe_median = []
    europe_upper_conf = []
    europe_lower_conf = []
    europe_stdev= []
    for obj in range(38):
        per_europe_obj_i = np.where(europe_all_targets==obj, 1, 0)
        med, std = bootstrap_ap(per_europe_obj_i, europe_all_scores[:, obj])
        europe_median.append(med * 100)
        europe_stdev.append(std)
        upper = (med.item() + (2 * std.item())) * 100
        lower = (med.item() - (2 * std.item())) * 100
        europe_upper_conf.append(upper)
        europe_lower_conf.append(lower)

    print("europe median")
    for num in europe_median:
        print(num)
    print()
    print("europe upper_conf")
    for num in europe_upper_conf:
        print(num)
    print()
    print("europe lower_conf")
    for num in europe_lower_conf:
        print(num)
    print()
    print("europe std without scaling by 100")
    for std in europe_stdev:
        print(std)
    print()

    print('XXXXXX')
    print()
    print("Accuracies")
    print()

    median_region, std_region = bootstrap_acc(region_all_targets, region_all_scores)
    median_europe, std_europe = bootstrap_acc(europe_all_targets, europe_all_scores)
    print("median region")
    print(median_region)
    print("upper_conf")
    print((median_region + (2 * std_region))*100)
    print("lower_conf")
    print((median_region - (2 * std_region))*100)
    print("std")
    print(std_region)
    print()
    print("median europe")
    print(median_europe)
    print("upper_conf")
    print((median_europe + (2 * std_europe))*100)
    print("lower_conf")
    print((median_europe - (2 * std_europe))*100)
    print("std")
    print(std_europe)

    with open('{}/europe_test_scores.pkl'.format(test_dir), 'wb+') as handle:
        pickle.dump({'scores':europe_all_scores, 'targets':europe_all_targets}, handle)
