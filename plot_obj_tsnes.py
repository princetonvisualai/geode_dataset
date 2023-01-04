import numpy as np
import pickle
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--reg', type=int, default=0, metavar='reg',
                    help='region number')
args = parser.parse_args()



def imscatter(dict_images, ax, zoom=1):
    #fig, ax = plt.subplots(figsize=(10, 10))
    #try:
    #    image = plt.imread(image)
    #except TypeError:
    #    # Likely already an array...
    #    pass
    #im = OffsetImage(image, zoom=zoom)
    #x, y = np.atleast_1d(x, y)
    artists = []
    for a in dict_images.keys():
        x = dict_images[a][0]
        y = dict_images[a][1]
        img = Image.open(a)
        w, h = img.size
        print(w,h)
        #scale = 2000/min(h,w)
        #img = img.resize((int(w*scale), int(h*scale)))
        im = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(im, (x, y), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
        ax.scatter(x, y)
        ax.update_datalim(np.array([x, y]).reshape(1, -1))
    ax.autoscale()
    #fig.savefig('tsne_appen/region{}.png'.format(reg))
    #fig.savefig('tsne_appen/region{}.eps'.format(reg))
    #fig.savefig('tsne_objects/obj{}.png'.format(reg))
    #fig.savefig('tsne_objects/obj{}.eps'.format(reg))

    return artists



np.set_printoptions(suppress=True)

#geoyfcc = pickle.load(open('data/geoyfcc_subsampled.pkl', 'rb'))

#data = geoyfcc['array']
#names = geoyfcc['names']


dict_appen = {}

for a in range(6):
    dict_appen[a] = pickle.load(open('data/appen/PASS_features/region{}.pkl'.format(a), 'rb'))


img_to_region = {}

per_region_features = {}
per_region_names = {}

for o in dict_appen[0].keys():
    per_region_features[o] = []
    per_region_names[o] = []
    for a in dict_appen.keys():
        for im in dict_appen[a][o]['train']:
            per_region_features[o].append(dict_appen[a][o]['train'][im].reshape(1, -1))
            per_region_names[o].append(im)
            img_to_region[im] = a

    per_region_features[o] = np.concatenate(per_region_features[o])

object_list = sorted(list(dict_appen[0]))

try:
    os.mkdir('pca_objects')
except:
    pass

for reg in [args.reg]: #range(len(object_list)):
    
    print(reg, object_list[reg]) 
    """
    import os

    if os.path.exists('tsne_objects/TSNE_obj{}.pkl'.format(reg)):
        p = pickle.load(open('tsne_objects/TSNE_obj{}.pkl'.format(reg), 'rb'))
        pca_a = p['pca']
        trans = p['trans']
        new_names = p['new_names']
    else:
    #if True:
        pca_a = PCA()
        new_data = pca_a.fit_transform(per_region_features[object_list[reg]].squeeze())
        print(pca_a.explained_variance_ratio_[:75].sum())
        pca_a = TSNE(n_components=2)
        rand_sample = np.random.choice(len(new_data), min(1000, len(new_data)), replace=False)
        new_data = new_data[rand_sample]
        new_names = [per_region_names[object_list[reg]][i] for i in rand_sample]

        trans = pca_a.fit_transform(new_data[:2500, :75])
        print('Done TSNE') 
        
        pickle.dump({'pca':pca_a, 'trans':trans, 'new_names':new_names}, open('tsne_objects/TSNE_obj{}.pkl'.format(reg), 'wb+'))

    sample = np.random.choice(len(trans), 200, replace=False)
    print(sample) 
    sample_dict = {}
    #print(names[0][:10])
    print(len(new_names), trans.shape)
    for i in sample:
        sample_dict[new_names[i]] = trans[i]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    imscatter(sample_dict, zoom=0.025)
    """
    """
    for a in dict_appen.keys():
        per_region_features[a] = []
        per_region_names[a] = []
        for o in dict_appen[a].keys():
            for im in dict_appen[a][o]['train']:
                per_region_features[a].append(dict_appen[a][o]['train'][im].reshape(1, -1))
                per_region_names[a].append(im)
        
        per_region_features[a] = np.concatenate(per_region_features[a])

    object_list = sorted(list(dict_appen[0]))
    for reg in range(6):
        
        import os

        if os.path.exists('tsne_appen/TSNE_reg{}.pkl'.format(reg)):
            p = pickle.load(open('tsne_appen/TSNE_reg{}.pkl'.format(reg), 'rb'))
            pca_a = p['pca']
            trans = p['trans']
            new_names = p['new_names']
        else:
        #if True:
            pca_a = PCA()
            new_data = pca_a.fit_transform(per_region_features[reg].squeeze())
            print(pca_a.explained_variance_ratio_[:75].sum())
            pca_a = TSNE(n_components=2)
            rand_sample = np.random.choice(len(new_data), 2500, replace=False)
            new_data = new_data[rand_sample]
            new_names = [per_region_names[reg][i] for i in rand_sample]

            trans = pca_a.fit_transform(new_data[:2500, :75])
            print('Done TSNE') 
            
            pickle.dump({'pca':pca_a, 'trans':trans, 'new_names':new_names}, open('tsne_appen/TSNE_reg{}.pkl'.format(reg), 'wb+'))

        sample = np.random.choice(len(trans), 200, replace=False)
        print(sample) 
        sample_dict = {}
        #print(names[0][:10])
        print(len(new_names), trans.shape)
        for i in sample:
            sample_dict[new_names[i]] = trans[i]
        
        fig, ax = plt.subplots(figsize=(10, 10))
        imscatter(sample_dict, zoom=0.1)
    """
    
    pca_a = PCA()
    trans = pca_a.fit_transform(per_region_features[object_list[reg]].squeeze())
    print(pca_a.explained_variance_ratio_[:100].sum()) 
    #tsne = TSNE(n_components=2)
    #trans = tsne.fit_transform(trans[:, :100])
    A = pickle.load(open('tsne_objects/mapping_{}.pkl'.format(reg), 'rb'))
    trans = A['points']
    
    print('Done TSNE') 

    """
    hist2d = np.histogram2d(trans[:, 0], trans[:, 1])
    print(hist2d[0])
    binx = hist2d[1]
    biny = hist2d[2]
    
    #min_y = int(input())
    #max_y = int(input())
    #min_x = int(input())
    #max_x = int(input())

    #hist2d = np.histogram2d(trans[:, 0], trans[:, 1], range=[[binx[min_x], binx[max_x]], [biny[min_y], biny[max_y]]])
    #print(hist2d[0])
    x_bins = hist2d[1]
    y_bins = hist2d[2]

    buckets = []
     
    for i in range(10):
        buckets.append([[] for a in range(10)])
    
    for i in range(len(trans)):
        if trans[i, 0]>x_bins[10]:
            continue
        if trans[i, 1]>y_bins[10]:
            continue
        for a in range(10):
            if trans[i, 0]<x_bins[a+1]:
                x_val = a
                break
        for a in range(10):
            if trans[i, 1]<y_bins[a+1]:
                y_val = a
                break
        buckets[x_val][y_val].append(per_region_names[object_list[reg]][i])
    """


    import matplotlib as mpl
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap

    cmap = mpl.cm.get_cmap('Dark2')
    fig, axs = plt.subplots(1, 1, figsize=(10,10))
    
    axs.scatter(trans[:, 0], trans[:, 1], c = [cmap(img_to_region[samp]) for samp in per_region_names[object_list[reg]]])
    #axs.scatter(trans[:, 0], trans[:, 1], c = cmap(2))
    
    sample = np.random.choice(len(trans), 20, replace=False)
    print(sample) 
    sample_dict = {}
    #print(names[0][:10])
    
    for i in sample:
        sample_dict[A['names'][i]] = trans[i]
    
    #imscatter(sample_dict, axs, zoom=0.05)

    """
    for i in range(10):
        for j in range(10):
            if len(buckets[i][j])==0:
                axs[i,j].axis('off')
                continue
            samp = np.random.choice(buckets[i][j], 1)[0]
            img = Image.open(samp)
            img_width, img_height = img.size
            crop_width = min(img_width, img_height)
            crop_height = crop_width


            new_img = img.crop(((img_width - crop_width)//2,
                                (img_height - crop_height)//2,
                                (img_width + crop_width)//2,
                                (img_height + crop_height)//2))
            axs[i, j].imshow(new_img)
            #axs[i,j].axis('off')
            axs[i, j].axes.xaxis.set_ticks([])
            axs[i, j].axes.yaxis.set_ticks([])
            
            for axis in ['top','bottom','left','right']:
                axs[i,j].spines[axis].set_linewidth(4)
                axs[i,j].spines[axis].set_color(cmap(img_to_region[samp]))
    """
    fig.savefig('tsne_objects/points_object{}.png'.format(reg), bbox_inches='tight')
    pickle.dump({'points':trans, 'names':per_region_names[object_list[reg]]}, open('tsne_objects/mapping_{}.pkl'.format(reg), 'wb+'))



