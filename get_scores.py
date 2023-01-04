import torch
from tqdm import tqdm
import torchvision
import model
import torchvision.transforms as T
import os
import argparse
from PIL import Image
import numpy as np
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=0)
args = vars(parser.parse_args())

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
dtype = torch.float32

arch = 'resnet50'

# for places365 model
model_file = '%s_places365.pth.tar' % arch
model = torchvision.models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
m = torch.nn.Sequential(*list(model.children())[:-1])


# for PASS trained model
m = torch.hub.load('yukimasano/PASS:main', 'swav_resnet50')

center_crop = T.Compose([
                T.Resize((256, 256)),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
m.eval()
m.cuda()


#imagenet features

dirs = 'data/im/'
objs = sorted([a for a in os.listdir(dirs) if not a.endswith('csv')])
for obj in objs:
    features = {}
    img_names = os.listdir('{}/{}'.format(dirs, obj))
    for i, img_name in enumerate(img_names):
        with torch.no_grad():    
            try:
                im = Image.open('{}/{}/{}'.format(dirs, obj, img_name).convert('RGB')

            except:
                continue
            X = center_crop(im).to(torch.device('cuda')).view(1, 3, 224, 224)
            scores= m(X)
            features[img] = scores.detach().cpu().numpy()

        if i%100==0:
            print(i, flush=True)
    
    with open('data/im/PASSi_features/{}.pkl'.format(obj), 'wb+') as handle:
        pickle.dump(features, handle)



## geode features

A = pickle.load(open('data/appen_prep_per_region.pkl', 'rb'))
reg = args['start']
features = {}
for obj in A[reg]:
    features[obj] = {}
    for split in ['train', 'test']:
        features[obj][split] = {}
        for i, img in enumerate(A[reg][obj][split]):
            with torch.no_grad():    
                try:
                    im = Image.open(img).convert('RGB')

                except:
                    continue
                X = center_crop(im).to(torch.device('cuda')).view(1, 3, 224, 224)
                scores= m(X)
                features[obj][split][img] = scores.detach().cpu().numpy()

            if i%100==0:
                print(i, flush=True)

with open('data/appen/PASS_features/region{}.pkl'.format(reg), 'wb+') as handle:
    pickle.dump(features, handle)
            

### Dollarstreet features
import json

categories = pickle.load(open('dollar_street_names_ids.pkl', 'rb'))
dollarstreet_info = json.load(open('Data/DollarStreet/dollar-street-images/info.json'))


features = {}

with torch.no_grad():    
    for y, name in categories:
        print(y, name)    
        for i in range(len(dollarstreet_info[name])):
            
            img_name = 'Data/DollarStreet/dollar-street-images/images/{}/{}.jpg'.format(name, dollarstreet_info[name][i]['img_id'])
            img = Image.open(img_name)
            X = center_crop(img).reshape(1, 3, 224, 224).to(torch.device('cuda'))
            
            scores = m(X).detach().cpu().numpy()
            
            country = dollarstreet_info[name][i]['country']
            features[img_name] = {'score':scores, 'target':name, 'country':country}



with open('dollarstreet_pass_feat.pkl', 'wb+') as handle:
    pickle.dump(features, handle)


## GeoYFCC with GeoDE tags

A = pickle.load(open('data/geoyfcc_with_country.pkl', 'rb'))

all_keys = sorted(A.keys())

features = {}

for i, a in enumerate(all_keys[args['start']:min(args['start']+10000, len(all_keys))]):
    
    full_path=a
    img = Image.open(full_path).convert('RGB')

    with torch.no_grad():
        X = center_crop(img).reshape(1, 3, 224, 224).to(torch.device('cuda'))
        scores = m(X).detach().cpu().numpy()

        features[full_path] = scores
    if i%1000==0:
        print(i, flush=True)

with open('data/geoyfcc_pass_feat/features{}.pkl'.format(args['start']), 'wb+') as handle:
    pickle.dump(features, handle)
