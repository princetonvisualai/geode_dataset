import torch
from os import mkdir, makedirs, path
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import numpy as np
from PIL import Image
import cv2
from basenet import *
import pandas as pd
import pickle
import load_data
import argparse
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

def compute_AP(target, predicted):
    per_class_AP = []
    for i in range(target.shape[1]):
        if not target[:, i].sum()==0: 
            per_class_AP.append(average_precision_score(target[:, i], predicted[:, i]))

    return np.mean(per_class_AP), per_class_AP

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--region_out', type=str, default='Africa')
    parser.add_argument('--train_batchsize', type=int, default=32)
    parser.add_argument('--val_batchsize', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--device', default=torch.device('cuda:0'))
    parser.add_argument('--dtype', default=torch.float32)
    parser.add_argument('--save', type=str, default='cifar/vgg')

    args = vars(parser.parse_args())
    

    print(type(args['region_out']), args['region_out'])
    model = ResNet50(n_out=1261, pretrained=args['pretrained'])
    if torch.cuda.is_available():
        model.cuda()
    
    model.train()
    model.require_all_grads()
    
    normalize = T.Normalize((0.4914,0.4822,0.4465),
                                     (0.2023,0.1994,0.2010))
    #normalize = T.Normalize(0.5, 0.5)

    transform_train = T.Compose([
        T.Resize((256, 256)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomCrop(224),
        T.ToTensor(),
        normalize,
    ])
    transform_test = T.Compose([
        T.Resize((256, 256)),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize,
    ])


    params = [
             {'batch_size': args['train_batchsize'],
             'shuffle': True,
             'num_workers': 0}, 
             
             {'batch_size': args['val_batchsize'],
             'shuffle': False,
             'num_workers': 0}
             ]
     
    optimizer_setting = {
        'optimizer': torch.optim.Adam,
        'lr': 1e-4,
    }
    optimizer = optimizer_setting['optimizer']( 
                        params=model.parameters(), 
                        lr=optimizer_setting['lr']) 
    count=0
    if torch.cuda.is_available(): 
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    dtype = torch.float32

    save_dir = 'record/'+args['save']
    if not path.exists(save_dir): 
        makedirs(save_dir)
    
    train_loader, val_loader, test_loader = load_data.create_geoyfcc_orig(load_data.Dataset, 
                                                                          params, 
                                                                          transform = [transform_train, transform_test], 
                                                                          region_out=args['region_out']) 

    best_AP = 0.0
    print(args['test'])
    if not args['test']:
        epoch_beg = 0
        if path.exists('{}/current.pth'.format(save_dir)):
            A = torch.load('{}/current.pth'.format(save_dir))
            model.load_state_dict(A['model'])
            optimizer.load_state_dict(A['optimizer'])
            epoch_beg = A['epoch']
        print('Training ... ', flush=True)
        for e in range(epoch_beg, args['epochs']):
            model.train()

            for t, (x, y) in enumerate(train_loader):
                x, y = x.to(device=device, dtype=dtype), y.to(device=device)
                #print(y, y.sum())
                scores = model(x)
                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(scores, y) 
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if t%100==0:
                    print(e, t, loss, flush=True, sep='\t')
            
            model.eval()
            correct = 0
            total = 0
            
            #cm_e = np.zeros((10, 10))
            all_scores = []
            all_targets = []
            with torch.no_grad():
                for X,Y in val_loader:
                    X = X.to(device=device, dtype=dtype)
                    Y = Y.to(device=device)

                    scores = torch.sigmoid(model(X))
                    all_scores.append(scores.cpu())
                    all_targets.append(Y.cpu())
            
            avg_AP, _ = compute_AP(np.concatenate(all_targets), np.concatenate(all_scores))
            print(e, t, avg_AP, flush=True)    
            
            if avg_AP>best_AP:
                best_AP =avg_AP
                torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch': e, 'loss':loss}, 
                            '{}/final.pth'.format(save_dir)
                            )

            torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch': e, 'loss':loss}, 
                        '{}/current.pth'.format(save_dir)
                        )

        

    A = torch.load('{}/final.pth'.format(save_dir))
    model.load_state_dict(A['model'])
    model.eval()
    
    correct = 0
    total = 0
     
    all_scores = []
    all_targets = []
    with torch.no_grad():
        for t, (X,Y) in enumerate(val_loader):
            X = X.to(device=device, dtype=dtype)
            Y = Y.to(device=device)

            scores = torch.sigmoid(model(X))
            all_scores.append(scores.cpu())
            all_targets.append(Y.cpu())
            if t%100==0:
                print(t, flush=True)
    avg_AP, APs = compute_AP(np.concatenate(all_targets), np.concatenate(all_scores))
    print(avg_AP, flush=True)    
    print(*APs, sep='\t')
    with open('{}/best_val_scores.pkl'.format(save_dir), 'wb+') as handle:
        pickle.dump({'targets':np.concatenate(all_targets), 'scores':np.concatenate(all_scores)}, handle)
    
    """
    extended_loader = load_data.create_openimages_crowdsourced_extended(load_data.Dataset, params, [transform_train, transform_test]) 
    all_scores = []
    all_targets = []
    with torch.no_grad():
        for t, (X,Y) in enumerate(extended_loader):
            X = X.to(device=device, dtype=dtype)
            Y = Y.to(device=device)

            scores = torch.sigmoid(model(X))
            all_scores.append(scores.cpu())
            all_targets.append(Y.cpu())
            if t%100==0:
                print(t, flush=True)
    avg_AP, APs = compute_AP(np.concatenate(all_targets), np.concatenate(all_scores))
    print(avg_AP, flush=True)    
    print(*APs, sep='\t')
    with open('{}/extended_scores.pkl'.format(save_dir), 'wb+') as handle:
        pickle.dump({'targets':np.concatenate(all_targets), 'scores':np.concatenate(all_scores)}, handle)
    """
    all_scores = []
    all_targets = []
    with torch.no_grad():
        for t, (X,Y) in enumerate(test_loader):
            X = X.to(device=device, dtype=dtype)
            Y = Y.to(device=device)

            scores = torch.sigmoid(model(X))
            all_scores.append(scores.cpu())
            all_targets.append(Y.cpu())
            if t%100==0:
                print(t, flush=True)
    avg_AP, APs = compute_AP(np.concatenate(all_targets), np.concatenate(all_scores))
    print(avg_AP, flush=True)    
    print(*APs, sep='\t')
    with open('{}/best_test_scores.pkl'.format(save_dir), 'wb+') as handle:
        pickle.dump({'targets':np.concatenate(all_targets), 'scores':np.concatenate(all_scores)}, handle)
