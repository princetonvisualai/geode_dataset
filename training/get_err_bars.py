import pickle
import utils
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--reg', type=int, default=0, metavar='reg',
                    help='region number')
parser.add_argument('--perc', type=float, default=0, metavar='reg',
                    help='percent to use for training')
parser.add_argument('--save', type=str, default='test', metavar='reg',
                    help='save directory')
parser.add_argument('--test', action='store_true')

args = parser.parse_args()

region_to_name = {0:'africa', 1:'americas', 2:'eastasia', 3:'europe', 4:'southeastasia', 5:'westasia'}


print('Region:{}, perc: {}'.format(region_to_name[args.reg], args.perc))

region_dset = pickle.load(open('/n/fs/vrdz-geodiv/singl/analysing_data/data/appen_prep_{}.pkl'.format(region_to_name[args.reg]), 'rb')) # change for region
europe_dset = pickle.load(open('/n/fs/vrdz-geodiv/singl/analysing_data/data/appen_prep_{}.pkl'.format('europe'), 'rb')) # change for region

reg_scores = pickle.load(open('{}/region{}_test_scores.pkl'.format(args.save, args.reg), 'rb'))
eur_scores = pickle.load(open('{}/europe_test_scores.pkl'.format(args.save), 'rb'))

reg_targets =  np.array(region_dset['test'][1])
eur_targets =  np.array(europe_dset['test'][1])
"""
print(reg_scores.shape, reg_targets.shape)

region_aps = []
europe_aps = []

region_std = []
europe_std = []


for i in range(38):
    
    curr_tar = np.where(reg_targets==i, 1, 0)

    med, std = utils.bootstrap_ap(curr_tar, reg_scores[:, i])
    print(i, med, std, sep='\t')
    region_aps.append(med)
    region_std.append(std)
    
with open('{}/region{}_obj_aps_std.pkl'.format(args.save, args.reg), 'wb+') as handle:
    pickle.dump({'ap':region_aps, 'std':region_std}, handle)


for i in range(38):
    
    curr_tar = np.where(eur_targets==i, 1, 0)

    med, std = utils.bootstrap_ap(curr_tar, eur_scores[:, i])
    print(i, med, std, sep='\t')
    
    europe_aps.append(med)
    europe_std.append(std)

with open('{}/europe_obj_aps_std.pkl'.format(args.save), 'wb+') as handle:
    pickle.dump({'ap':europe_aps, 'std':europe_std}, handle)

"""
med, std = utils.bootstrap_acc(reg_targets, reg_scores)
print('region', 100*med, 100*std, sep=', ')

med, std = utils.bootstrap_acc(eur_targets, eur_scores)
print('europe', 100*med, 100*std, sep=', ')

