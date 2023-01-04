from __future__ import print_function, division
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import torchvision.transforms as T
import pickle
import csv
from sklearn.model_selection import train_test_split
from os import listdir
import pickle
import pandas as pd

class Dataset(Dataset):
    def __init__(self, img_paths, img_labels, transform=T.ToTensor(), output_size=30):
        print(len(img_paths))
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.transform = transform
        self.output_size = output_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        ID = self.img_paths[index]
        img = Image.open(ID).convert('RGB')
        X = self.transform(img)
        y = torch.zeros(self.output_size)
        y[self.img_labels[ID]] = 1
        return X, y



def create_geoyfcc_orig(dataset, params, region_out='Africa', transform=None):

    df = pickle.load(open('Data/GeoYFCC/subset_geoyfcc0.pkl', 'rb'))
    for i in range(100000, 1200000, 100000):
        df2 = pickle.load(open('Data/GeoYFCC/subset_geoyfcc{}.pkl'.format(i), 'rb'))
        df = pd.concat([df, df2])

    bad_names = ['Data/GeoYFCC/images/c06/6b5/c066b528fbcd5ec132b878681da8c14.jpg', 
                 'Data/YFCC100m/data/images/c76/13c/c7613c78acb196d985d46e2f9972ac.jpg']
    row_to_image_ids = pickle.load(open('Data/GeoYFCC/row_to_image.pkl', 'rb'))
    img_id_to_hash = pickle.load(open('Data/GeoYFCC/imgid_hash.pkl', 'rb'))

    locations = pickle.load(open('Data/GeoYFCC/image_id_to_location0.pkl', 'rb'))
    for i in range(100000, 1200000, 100000):
        loc = pickle.load(open('Data/GeoYFCC/image_id_to_location{}.pkl'.format(i), 'rb'))
        locations.update(loc)

    africa = ['Kenya', 'Tanzania', 'Morocco', 'Egypt', 'South Africa']
    asia = ['Indonesia', 'United Arab Emirates', 'India', 'Singapore', 'Jordan', 'Cambodia', 
            'China', 'Philippines', 'South Korea', 'Vietnam', 'Israel', 'Taiwan', 'Japan',
            'Malaysia', 'Nepal', 'Thailand', 'Turkey','Russia']
    europe = ['Bulgaria', 'United Kingdom', 'Croatia', 'Norway', 'Poland', 'Portugal', 'Russia', 
              'Czech Republic', 'Denmark', 'Finland', 'Germany', 'Spain', 'Switzerland', 'Greece', 
              'Hungary', 'Iceland', 'Ireland', 'Turkey', 'Ukraine', 'Romania', 'Italy', 
              'Netherlands', 'France', 'Sweden', 'Austria', 'Belgium']
    south_america = ['Brazil', 'Colombia', 'Peru', 'Argentina', 'Chile']
    north_america = ['Canada', 'United States', 'Panama', 'Mexico', 'Costa Rica', 'Cuba', 'The Bahamas']
    oceania=['Australia', 'New Zealand']

    
    per_region_img_ids = {'Africa':[], 
                          'Europe':[],
                          'Asia':[],
                          'NorthAmerica':[],
                          'SouthAmerica':[],
                          'Oceania':[]}
    labels = {}

    for t in df.index:
        #print(i)
        row_num = int(df['yfcc_row_id'][t])
        img_id = int(row_to_image_ids[row_num])

        full_img_name = locations[img_id]
        if full_img_name in bad_names:
            continue
        if df['country'][t] in africa:
            per_region_img_ids['Africa'].append(full_img_name)
        elif df['country'][t] in europe:
            per_region_img_ids['Europe'].append(full_img_name)
        elif df['country'][t] in asia:
            per_region_img_ids['Asia'].append(full_img_name)
        elif df['country'][t] in north_america:
            per_region_img_ids['NorthAmerica'].append(full_img_name)
        elif df['country'][t] in south_america:
            per_region_img_ids['SouthAmerica'].append(full_img_name)
        elif df['country'][t] in oceania:
            per_region_img_ids['Oceania'].append(full_img_name)

        labels[full_img_name] = df['label_ids'][t]
    

    train_ids = []
    for key in per_region_img_ids:
        if key!=region_out:
            train_ids+=per_region_img_ids[key]

    
    train_set, val_set = train_test_split(train_ids, train_size=0.8, random_state=42) 
    
    #if len(train_set)>100000:
    #    train_set = train_set[np.random.choice(len(train_set), 100000)]

    test_set = per_region_img_ids[region_out]
    print(len(train_set), len(val_set), len(test_set))

    #val_set, test_set = train_test_split(valtest_set, train_size=0.5, random_state=42) 
        
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    if transform==None: 
        transform = [
            T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize
            ]), 
            T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize
            ])] 

       
    #print(transform)    
    dset_train = dataset(train_set, labels, transform[0], output_size=1261)
    loader_train = DataLoader(dset_train, **params[0])
    
    dset_val = dataset(val_set, labels, transform[1], output_size=1261)
    loader_val = DataLoader(dset_val, **params[1])

    dset_test = dataset(test_set, labels, transform[1], output_size=1261)
    loader_test = DataLoader(dset_test, **params[1])
    
    return loader_train, loader_val, loader_test

def create_geoyfcc_frac_region(dataset, params, region_out='Africa', frac=0.1, transform=None):

    df = pickle.load(open('Data/GeoYFCC/subset_geoyfcc0.pkl', 'rb'))
    for i in range(100000, 1200000, 100000):
        df2 = pickle.load(open('Data/GeoYFCC/subset_geoyfcc{}.pkl'.format(i), 'rb'))
        df = pd.concat([df, df2])

    bad_names = ['Data/GeoYFCC/images/c06/6b5/c066b528fbcd5ec132b878681da8c14.jpg', 
                 'Data/YFCC100m/data/images/c76/13c/c7613c78acb196d985d46e2f9972ac.jpg']
    row_to_image_ids = pickle.load(open('Data/GeoYFCC/row_to_image.pkl', 'rb'))
    img_id_to_hash = pickle.load(open('Data/GeoYFCC/imgid_hash.pkl', 'rb'))

    locations = pickle.load(open('Data/GeoYFCC/image_id_to_location0.pkl', 'rb'))
    for i in range(100000, 1200000, 100000):
        loc = pickle.load(open('Data/GeoYFCC/image_id_to_location{}.pkl'.format(i), 'rb'))
        locations.update(loc)

    africa = ['Kenya', 'Tanzania', 'Morocco', 'Egypt', 'South Africa']
    asia = ['Indonesia', 'United Arab Emirates', 'India', 'Singapore', 'Jordan', 'Cambodia', 
            'China', 'Philippines', 'South Korea', 'Vietnam', 'Israel', 'Taiwan', 'Japan',
            'Malaysia', 'Nepal', 'Thailand', 'Turkey','Russia']
    europe = ['Bulgaria', 'United Kingdom', 'Croatia', 'Norway', 'Poland', 'Portugal', 'Russia', 
              'Czech Republic', 'Denmark', 'Finland', 'Germany', 'Spain', 'Switzerland', 'Greece', 
              'Hungary', 'Iceland', 'Ireland', 'Turkey', 'Ukraine', 'Romania', 'Italy', 
              'Netherlands', 'France', 'Sweden', 'Austria', 'Belgium']
    south_america = ['Brazil', 'Colombia', 'Peru', 'Argentina', 'Chile']
    north_america = ['Canada', 'United States', 'Panama', 'Mexico', 'Costa Rica', 'Cuba', 'The Bahamas']
    oceania=['Australia', 'New Zealand']

    
    per_region_img_ids = {'Africa':[], 
                          'Europe':[],
                          'Asia':[],
                          'NorthAmerica':[],
                          'SouthAmerica':[],
                          'Oceania':[]}
    labels = {}

    for t in df.index:
        #print(i)
        row_num = int(df['yfcc_row_id'][t])
        img_id = int(row_to_image_ids[row_num])

        full_img_name = locations[img_id]
        if full_img_name in bad_names:
            continue
        if df['country'][t] in africa:
            per_region_img_ids['Africa'].append(full_img_name)
        elif df['country'][t] in europe:
            per_region_img_ids['Europe'].append(full_img_name)
        elif df['country'][t] in asia:
            per_region_img_ids['Asia'].append(full_img_name)
        elif df['country'][t] in north_america:
            per_region_img_ids['NorthAmerica'].append(full_img_name)
        elif df['country'][t] in south_america:
            per_region_img_ids['SouthAmerica'].append(full_img_name)
        elif df['country'][t] in oceania:
            per_region_img_ids['Oceania'].append(full_img_name)

        labels[full_img_name] = df['label_ids'][t]
    

    train_ids = []
    for key in per_region_img_ids:
        if key!=region_out:
            train_ids+=per_region_img_ids[key]

    #region_out_size = len(per_region_img_ids[region_out])
    
    region_out_train, region_out_test = train_test_split(per_region_img_ids[region_out], train_size=0.8, random_state=42)
    train_size = len(region_out_train)


    train_set, val_set = train_test_split(train_ids, train_size=int((1-frac)*train_size), random_state=42) 
    
    region_out_train, region_out_val = train_test_split(region_out_train, train_size = int(frac*train_size), random_state=42)
    
    train_set = train_set+region_out_train
    val_set = val_set + region_out_val
    
    #if len(train_set)>100000:
    #    train_set = train_set[np.random.choice(len(train_set), 100000)]

    test_set = region_out_test
    print(len(train_set), len(val_set), len(test_set))

    #val_set, test_set = train_test_split(valtest_set, train_size=0.5, random_state=42) 
        
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    if transform==None: 
        transform = [
            T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize
            ]), 
            T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize
            ])] 

       
    #print(transform)    
    dset_train = dataset(train_set, labels, transform[0], output_size=1261)
    loader_train = DataLoader(dset_train, **params[0])
    
    dset_val = dataset(val_set, labels, transform[1], output_size=1261)
    loader_val = DataLoader(dset_val, **params[1])

    dset_test = dataset(test_set, labels, transform[1], output_size=1261)
    loader_test = DataLoader(dset_test, **params[1])
    
    return loader_train, loader_val, loader_test

def create_geoyfcc_country(dataset, params, region_out='Africa', transform=None):
    
    df = pickle.load(open('Data/GeoYFCC/subset_geoyfcc0.pkl', 'rb'))
    for i in range(100000, 1200000, 100000):
        df2 = pickle.load(open('Data/GeoYFCC/subset_geoyfcc{}.pkl'.format(i), 'rb'))
        df = pd.concat([df, df2])

    bad_names = ['Data/GeoYFCC/images/c06/6b5/c066b528fbcd5ec132b878681da8c14.jpg', 
                 'Data/YFCC100m/data/images/c76/13c/c7613c78acb196d985d46e2f9972ac.jpg']
    row_to_image_ids = pickle.load(open('Data/GeoYFCC/row_to_image.pkl', 'rb'))
    img_id_to_hash = pickle.load(open('Data/GeoYFCC/imgid_hash.pkl', 'rb'))

    locations = pickle.load(open('Data/GeoYFCC/image_id_to_location0.pkl', 'rb'))
    for i in range(100000, 1200000, 100000):
        loc = pickle.load(open('Data/GeoYFCC/image_id_to_location{}.pkl'.format(i), 'rb'))
        locations.update(loc)


    africa = ['Kenya', 'Tanzania', 'Morocco', 'Egypt', 'South Africa']
    asia = ['Indonesia', 'United Arab Emirates', 'India', 'Singapore', 'Jordan', 'Cambodia', 
            'China', 'Philippines', 'South Korea', 'Vietnam', 'Israel', 'Taiwan', 'Japan',
            'Malaysia', 'Nepal', 'Thailand', 'Turkey','Russia']
    europe = ['Bulgaria', 'United Kingdom', 'Croatia', 'Norway', 'Poland', 'Portugal', 'Russia', 
              'Czech Republic', 'Denmark', 'Finland', 'Germany', 'Spain', 'Switzerland', 'Greece', 
              'Hungary', 'Iceland', 'Ireland', 'Turkey', 'Ukraine', 'Romania', 'Italy', 
              'Netherlands', 'France', 'Sweden', 'Austria', 'Belgium']
    south_america = ['Brazil', 'Colombia', 'Peru', 'Argentina', 'Chile']
    north_america = ['Canada', 'United States', 'Panama', 'Mexico', 'Costa Rica', 'Cuba', 'The Bahamas']
    oceania=['Australia', 'New Zealand']

    
    per_region_img_ids = {'Africa':[], 
                          'Europe':[],
                          'Asia':[],
                          'NorthAmerica':[],
                          'SouthAmerica':[],
                          'Oceania':[]}
    labels = {}
    country_labels = {}
    for t in df.index:
        #print(i)
        row_num = int(df['yfcc_row_id'][t])
        img_id = int(row_to_image_ids[row_num])

        full_img_name = locations[img_id]
        if full_img_name in bad_names:
            continue
        if df['country'][t] in africa:
            per_region_img_ids['Africa'].append(full_img_name)
        elif df['country'][t] in europe:
            per_region_img_ids['Europe'].append(full_img_name)
        elif df['country'][t] in asia:
            per_region_img_ids['Asia'].append(full_img_name)
        elif df['country'][t] in north_america:
            per_region_img_ids['NorthAmerica'].append(full_img_name)
        elif df['country'][t] in south_america:
            per_region_img_ids['SouthAmerica'].append(full_img_name)
        elif df['country'][t] in oceania:
            per_region_img_ids['Oceania'].append(full_img_name)

        labels[full_img_name] = df['label_ids'][t]
        country_labels[full_img_name] = df['country'][t]


    train_ids = []
    for key in per_region_img_ids:
        if key!=region_out:
            train_ids+=per_region_img_ids[key]

    
    train_set, val_set = train_test_split(train_ids, train_size=0.8, random_state=42) 
    test_set = per_region_img_ids[region_out]
        
    return val_set, test_set, country_labels 

def create_geoyfcc_separate_region(dataset, params, region_out='Africa', transform=None, all_regions = False):


    df = pickle.load(open('Data/GeoYFCC/subset_geoyfcc0.pkl', 'rb'))
    for i in range(100000, 1200000, 100000):
        df2 = pickle.load(open('Data/GeoYFCC/subset_geoyfcc{}.pkl'.format(i), 'rb'))
        df = pd.concat([df, df2])

    bad_names = ['Data/GeoYFCC/images/c06/6b5/c066b528fbcd5ec132b878681da8c14.jpg', 
                 'Data/YFCC100m/data/images/c76/13c/c7613c78acb196d985d46e2f9972ac.jpg']
    row_to_image_ids = pickle.load(open('Data/GeoYFCC/row_to_image.pkl', 'rb'))
    img_id_to_hash = pickle.load(open('Data/GeoYFCC/imgid_hash.pkl', 'rb'))

    locations = pickle.load(open('Data/GeoYFCC/image_id_to_location0.pkl', 'rb'))
    for i in range(100000, 1200000, 100000):
        loc = pickle.load(open('Data/GeoYFCC/image_id_to_location{}.pkl'.format(i), 'rb'))
        locations.update(loc)

    africa = ['Kenya', 'Tanzania', 'Morocco', 'Egypt', 'South Africa']
    asia = ['Indonesia', 'United Arab Emirates', 'India', 'Singapore', 'Jordan', 'Cambodia', 
            'China', 'Philippines', 'South Korea', 'Vietnam', 'Israel', 'Taiwan', 'Japan',
            'Malaysia', 'Nepal', 'Thailand', 'Turkey','Russia']
    europe = ['Bulgaria', 'United Kingdom', 'Croatia', 'Norway', 'Poland', 'Portugal', 'Russia', 
              'Czech Republic', 'Denmark', 'Finland', 'Germany', 'Spain', 'Switzerland', 'Greece', 
              'Hungary', 'Iceland', 'Ireland', 'Turkey', 'Ukraine', 'Romania', 'Italy', 
              'Netherlands', 'France', 'Sweden', 'Austria', 'Belgium']
    south_america = ['Brazil', 'Colombia', 'Peru', 'Argentina', 'Chile']
    north_america = ['Canada', 'United States', 'Panama', 'Mexico', 'Costa Rica', 'Cuba', 'The Bahamas']
    oceania=['Australia', 'New Zealand']

    
    per_region_img_ids = {'Africa':[], 
                          'Europe':[],
                          'Asia':[],
                          'NorthAmerica':[],
                          'SouthAmerica':[],
                          'Oceania':[]}
    labels = {}
    country_labels = {}
    for t in df.index:
        #print(i)
        row_num = int(df['yfcc_row_id'][t])
        img_id = int(row_to_image_ids[row_num])

        full_img_name = locations[img_id]
        if full_img_name in bad_names:
            continue
        if df['country'][t] in africa:
            per_region_img_ids['Africa'].append(full_img_name)
        elif df['country'][t] in europe:
            per_region_img_ids['Europe'].append(full_img_name)
        elif df['country'][t] in asia:
            per_region_img_ids['Asia'].append(full_img_name)
        elif df['country'][t] in north_america:
            per_region_img_ids['NorthAmerica'].append(full_img_name)
        elif df['country'][t] in south_america:
            per_region_img_ids['SouthAmerica'].append(full_img_name)
        elif df['country'][t] in oceania:
            per_region_img_ids['Oceania'].append(full_img_name)

        labels[full_img_name] = df['label_ids'][t]
        country_labels[full_img_name] = df['country'][t]

    if all_regions:
        image_set = []
        for reg in per_region_img_ids.keys():
            image_set+=per_region_img_ids[reg]
    else:
        image_set = per_region_img_ids[region_out]
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    if transform==None: 
        transform = T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize
            ])

    #dset_region = dataset(image_set, labels, transform, output_size=1261)
    #loader_region = DataLoader(dset_region, **params)
    return image_set, labels, country_labels 

if __name__=='__main__':
    
    params = {'batch_size': 256,
             'shuffle': True,
             'num_workers': 0}
