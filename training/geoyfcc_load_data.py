import utils_local
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import torchvision.transforms as T
import numpy as np

def prep_geoyfcc_region():
   obj_mappings = {'0':[149, 150, 151, 152, 278],
                   '1':[194, 195, 196, 611, 896, 904, 916],
                   '2':[272, 288, 375, 430, 474, 531, 554, 559, 607, 628, 740, 752, 769, 805, 927],
                   '3':[254, 596, 699, 906],
                   '4':[256, 275, 349, 759, 810],
                   '5':[301, 303, 361, 387, 456, 831, 876],
                   '6':[242, 243, 244, 509],
                   '7':[229, 395, 643, 964, 987],
                   '8':[132],
                   '9':[428, 661],
                   '10':[443],
                   '11':[404],
                   '12':[211, 230, 324, 426, 497, 671, 801],
                   '13':[117, 245, 250, 512, 513],
                   '14':[545, 547, 548, 556, 305],
                   '15':[557],
                   '16':[95, 125, 126, 241, 286, 289, 320, 466, 481, 589, 595, 626, 642, 735, 819, 274],
                   '17':[482, 675, 975, 976, 977, 978, 979, 980, 982, 983, 988, 994, 995, 996, 999, 
                     1004, 1005, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1015, 1016, 1058, 
                     1069, 1073, 1074, 1075, 1076, 1077],
                   '18':[108, 306, 315, 600, 605, 822, 838, 870],
                   '19':[1021, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1224],
                   '20':[764, 813, 818],
                   '21':[486, 775],
                   '22':[832, 833],
                   '23':[546, 836],
                   '24':[400, 893],
                   '25':[1180, 1181, 1182, 1183, 1228, 1231, 1233, 1239, 1241, 1242, 1247],
                   '26':[662, 854, 895, 897, 898, 919],
                   '27':[411, 464],
                   '28':[168]}

   geoyfcc = pickle.load(open('data/images_present_geoyfcc.pkl', 'rb'))
   all_countries = ["Egypt", "Nigeria", "South Africa", "Kenya", "Egypy", "Tanzania", "Morocco", "Argentina", "Colombia", 
                    "Mexico", "Brazil", "Peru", "Costa Rica", "Panama", "Chile", "China", "Japan", "South Korea", 
                    "Singapore", "Taiwan", "Indonesia", "Philippines", "Thailand", "Vietnam", "Cambodia", "Malaysia", 
                    "Saudi Arabia", "United Arab Emirates", "Turkey", "Jordan", "Israel", "Italy", "Romania", "Spain", 
                    "United Kingdom", "Switzerland", "Ireland", "Iceland", "Bulgaria", "Sweden", "Croatia", "Hungary", 
                    "Ukraine", "Belgium", "Austria", "Norway", "Denmark", "Germany", "Poland", "France", "Czech Republic", 
                    "Netherlands", "Finland", "Greece", "Portugal"]

   df_allcountries = geoyfcc
   print(len(all_countries))

   all_obj_idx = []
   for obj in range(29):
      for obj_indx in obj_mappings[str(obj)]:
         all_obj_idx.append(obj_indx)
   
   row_id = set()
   for index, row in df_allcountries.iterrows():
      for obj_id in row['label_ids']:
         if obj_id in all_obj_idx:
            row_id.add(row['yfcc_row_id'])
   df_allcountries_30obj = df_allcountries[df_allcountries.yfcc_row_id.isin(row_id)]

   for index, row in df_allcountries_30obj.iterrows():
      new_indices = set()
      for obj_id in row['label_ids']:
         for obj in range(29):
            if obj_id in obj_mappings[str(obj)]:
               new_indices.add(int(obj))

      df_allcountries_30obj.at[index, 'label_ids'] = list(new_indices)

   africa = ["Egypt", "Nigeria", "South Africa", "Kenya", "Egypy", "Tanzania", "Morocco"]
   americas = ["Argentina", "Colombia", "Mexico", "Brazil", "Peru", "Costa Rica", "Panama", "Chile"]
   eastasia = ["China", "Japan", "South Korea", "Singapore", "Taiwan"]
   southeastasia = ["Indonesia", "Philippines", "Thailand", "Vietnam", "Cambodia", "Malaysia"]
   westasia = ["Saudi Arabia", "United Arab Emirates", "Turkey", "Jordan", "Israel"]
   europe = ["Italy", "Romania", "Spain", "United Kingdom", "Switzerland", "Ireland", "Iceland", "Bulgaria", "Sweden",
             "Croatia", "Hungary", "Ukraine", "Belgium", "Austria", "Norway", "Denmark", "Germany", "Poland", "France",
             "Czech Republic", "Netherlands", "Finland", "Greece", "Portugal"]

   df_africa = df_allcountries_30obj[df_allcountries_30obj.country.isin(africa)]
   df_americas = df_allcountries_30obj[df_allcountries_30obj.country.isin(americas)]
   df_eastasia = df_allcountries_30obj[df_allcountries_30obj.country.isin(eastasia)]
   df_southeastasia = df_allcountries_30obj[df_allcountries_30obj.country.isin(southeastasia)]
   df_westasia = df_allcountries_30obj[df_allcountries_30obj.country.isin(westasia)]
   df_europe = df_allcountries_30obj[df_allcountries_30obj.country.isin(europe)]

   yfcc_row_id = set(df_allcountries_30obj["yfcc_row_id"])
   yfcc_row_id = list(yfcc_row_id)

   row_to_imageid = pickle.load(open('.../GeoYFCC/row_to_image.pkl', 'rb'))

   image_id_to_location0 = pickle.load(open('.../GeoYFCC/image_id_to_location0.pkl', 'rb'))
   image_id_to_location100000 = pickle.load(open('.../GeoYFCC/image_id_to_location100000.pkl', 'rb'))
   image_id_to_location200000 = pickle.load(open('.../GeoYFCC/image_id_to_location200000.pkl', 'rb'))
   image_id_to_location300000 = pickle.load(open('.../GeoYFCC/image_id_to_location300000.pkl', 'rb'))
   image_id_to_location400000 = pickle.load(open('.../GeoYFCC/image_id_to_location400000.pkl', 'rb'))
   image_id_to_location500000 = pickle.load(open('.../GeoYFCC/image_id_to_location500000.pkl', 'rb'))
   image_id_to_location600000 = pickle.load(open('.../GeoYFCC/image_id_to_location600000.pkl', 'rb'))
   image_id_to_location700000 = pickle.load(open('.../GeoYFCC/image_id_to_location700000.pkl', 'rb'))
   image_id_to_location800000 = pickle.load(open('.../GeoYFCC/image_id_to_location800000.pkl', 'rb'))
   image_id_to_location900000 = pickle.load(open('.../GeoYFCC/image_id_to_location900000.pkl', 'rb'))
   image_id_to_location1000000 = pickle.load(open('.../GeoYFCC/image_id_to_location1000000.pkl', 'rb'))
   image_id_to_location1100000 = pickle.load(open('.../GeoYFCC/image_id_to_location1100000.pkl', 'rb'))

   image_id_to_location = {**image_id_to_location0, **image_id_to_location100000, **image_id_to_location200000,
                           **image_id_to_location300000, **image_id_to_location400000, **image_id_to_location500000,
                           **image_id_to_location600000, **image_id_to_location700000, **image_id_to_location800000,
                           **image_id_to_location900000, **image_id_to_location1000000, **image_id_to_location1100000}
   
   image_names = []
   obj = []

   for row_id in yfcc_row_id:
      image_id = row_to_imageid[np.int64(row_id)]
      location = image_id_to_location[np.int64(image_id)]
      image_names.append(location)
      idx = np.int64(df_allcountries_30obj.index[df_allcountries_30obj['yfcc_row_id'] == str(row_id)])[0]
      labels = df_allcountries_30obj.at[idx, 'label_ids']
      obj.append(labels)
   
   train_names, valtest_names, train_obj, valtest_obj = train_test_split(image_names, obj, random_state = 42, test_size=0.4)
   val_names, test_names, val_obj, test_obj = train_test_split(valtest_names, valtest_obj, random_state = 42, test_size=0.5)

   if not os.path.exists('data'):     
        os.mkdir('data')
   with open('data/geoyfcc_prep.pkl', 'wb+') as handle:
        pickle.dump({'train':[train_names, train_obj], 
                     'val':[val_names, val_obj], 
                     'test':[test_names, test_obj]}, handle)

def imagenet_prep():
   master_csv = pd.read_csv('data/imagenet40_filenames.csv')
   categories = ["bag", "bicycle", "boat", "bus", "car", "chair", "cleaning_equipment", "cooking_pot", "dustbin",
          "fence", "flag", "front_door", "hat", "house", "light_fixture", "lighter", "monument",
          "plate_of_food", "religious_building", "spices", "stall", "storefront", "stove", "streetlight_lattern",
          "toy", "tree", "truck", "waste_container", "wheelbarrow"]

   imagenet29 = master_csv[master_csv.script_name.isin(categories)]
   print(imagenet29)

   image_names = [] # stores directory path to each image
   obj = []

   all_obj_names = sorted(list(imagenet29['script_name'].unique()))
   print(all_obj_names)
   print("num of categories: " + str(len(all_obj_names)))

   for idx in imagenet29.index: # for each image in imagenet29
      fname = imagenet29['file_name'][idx] # country_category_#.jpg
      oname = imagenet29['script_name'][idx] # object category of the image
      image_names.append('.../{}/{}'.format(oname, fname))
      lst_obj = []
      lst_obj.append(all_obj_names.index(oname))
      obj.append(lst_obj) # number label corresponding to the object classification

   needed_training = 27620
   imagenet_num_images_ratio = needed_training/len(image_names)
   imagenet_random_sample = []

   for category in range(29):
      sum = 0
      subset = []
      for idx in range(len(obj)):
         if category == obj[idx][0]:
            sum = sum + 1
            subset.append(idx)
      num_images = np.floor(imagenet_num_images_ratio*sum).astype(int)
      subset_sample = np.random.choice(subset, num_images, replace=False)
      imagenet_random_sample = [*imagenet_random_sample, *subset_sample]
   
   needed_training -= len(imagenet_random_sample)
   full_indices = np.arange(len(image_names))
   excluded_pics = np.array([i for i in full_indices if i not in imagenet_random_sample])
   print("number excluded: " + str(len(excluded_pics)))
   print("number of images before sample: " + str(len(image_names)))
   add_on = np.random.choice(excluded_pics, needed_training, replace=False)
   print("number of add ons: " + str(len(add_on)))
   imagenet_random_sample = [*imagenet_random_sample, *add_on]
   
   imagenet_names = [image_names[i] for i in imagenet_random_sample]
   imagenet_obj = [obj[i] for i in imagenet_random_sample]
   print("number of images: " + str(len(imagenet_names)))

   train_names, valtest_names, train_obj, valtest_obj = train_test_split(imagenet_names, imagenet_obj, random_state = 42, test_size=0.4)
   val_names, test_names, val_obj, test_obj = train_test_split(valtest_names, valtest_obj, random_state = 42, test_size=0.5)

   print("number of training images: " + str(len(train_names)))

   '''
   if not os.path.exists('data'):     
      os.mkdir('data')
   with open('data/imagenet29_prep.pkl', 'wb+') as handle:
      pickle.dump({'train':[train_names, train_obj], 
                   'val':[val_names, val_obj], 
                   'test':[test_names, test_obj]}, handle)
   '''

def prep_geode():
   master_csv = pd.read_csv('data/geode38_meta.csv')

   categories = ["Bag Image", "Bicycle Image", "Boat Image", "Bus Image", "Car Image", "Chair Image", "Cleaning equipment Image", "Cooking pot Image", "Dustbin Image",
          "Fence Image", "Flag Image", "Front door Image", "Hat Image", "House Image", "Light fixture Image", "Lighter Image", "Monument Image",
          "Plate of food Image", "Religious building Image", "Spices Image", "Stall Image", "Storefront Image", "Stove Image", "Streetlight lantern Image",
          "Toy Image", "Tree Image", "Truck Image", "Waste container Image", "Wheelbarrow Image"]

   geode29 = master_csv[master_csv.script_name.isin(categories)]

   image_names = [] # stores directory path to each image
   obj = []

   country_to_region = utils_local.get_country_to_region() # maps country:number
   region_to_number = utils_local.get_reg_to_number() # maps region name:region number
   number_to_region = {v:k for (k,v) in region_to_number.items()} # changes map to number: region

   all_obj_names = sorted(list(geode29['script_name'].unique()))
   print(all_obj_names)
   print("num of categories: " + str(len(all_obj_names)))

   for idx in geode29.index: # for each image in geode29
      fname = geode29['file_name'][idx] # country_category_#.jpg
      oname = geode29['script_name'][idx] # object category of the image
      cname = geode29['ip_country'][idx].replace(' ', '_') # country of the image
      region = number_to_region[country_to_region[cname]]
      image_names.append('.../{}/{}/{}'.format(
                                                                    number_to_region[country_to_region[cname]], 
                                                                    cname, fname))
      lst_obj = []
      lst_obj.append(all_obj_names.index(oname))
      obj.append(lst_obj) # number label corresponding to the object classification

   all_indices = np.arange(len(image_names))
   geode_random_sample = np.random.choice(all_indices, 36303, replace=False)
   geode_names = [image_names[i] for i in geode_random_sample]
   geode_obj = [obj[i] for i in geode_random_sample]

   train_names, valtest_names, train_obj, valtest_obj = train_test_split(geode_names, geode_obj, random_state = 42, test_size=0.4)
   val_names, test_names, val_obj, test_obj = train_test_split(valtest_names, valtest_obj, random_state = 42, test_size=0.5)
   print("number of training images: " + str(len(train_names)))

   if not os.path.exists('data'):     
      os.mkdir('data')
   with open('data/geode29_prep.pkl', 'wb+') as handle:
      pickle.dump({'train':[train_names, train_obj], 
                   'val':[val_names, val_obj], 
                   'test':[test_names, test_obj]}, handle)


if __name__=="__main__":
   #imagenet_prep()
   prep_geoyfcc_region()
   #prep_geode()
