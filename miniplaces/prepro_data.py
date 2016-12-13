import os
import numpy as np
import h5py
import scipy.misc
import pickle
import json

def list5(tp):
	ret = list(tp)[:5]
	while len(ret) <5:
		ret.append(-1)
	return ret

def createH5(params):

	# create output h5 file
	output_h5 = '%s_%d_%s.h5' %(params['name'], params['img_resize'], params['split'])
	f_h5 = h5py.File(output_h5, "w")

	# read data info from lists
	list_im = []
	list_lab = []
	tag_list = pickle.load(open(params["tag_list"],"r"))
	list_tag = []
	with open(params['data_list'], 'r') as f:
	    for line in f:
	        path, lab =line.rstrip().split(' ')
	        list_im.append(os.path.join(params['data_root'], path))
	        list_lab.append(int(lab))
		list_tag.append(list5(tag_list.get(list_im[-1],tuple([]))))
	list_im = np.array(list_im, np.object)
	list_lab = np.array(list_lab, np.uint16)
	list_tag = np.array(list_tag, np.object)
	print(list_tag.shape)
	N = list_im.shape[0]
	print('# Images found:'), N
	
	# permutation
	perm = np.random.permutation(N) 
	list_im = list_im[perm]
	list_lab = list_lab[perm]
	list_tag = list_tag[perm]

	im_set = f_h5.create_dataset("images", (N,params['img_resize'],params['img_resize'],3), dtype='uint8') # space for resized images
	f_h5.create_dataset("labels", dtype='uint16', data=list_lab)
	f_h5.create_dataset("tags",   dtype='int16', data=list_tag)

	for i in range(N):
		image = scipy.misc.imread(list_im[i])
		assert image.shape[2]==3, 'Channel size error!'
		image = scipy.misc.imresize(image, (params['img_resize'],params['img_resize']))

		im_set[i] = image

		if i % 1000 == 0:
			print 'processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N)

	f_h5.close()

if __name__=='__main__':
	params_train = {
		'name': 'miniplaces2',
		'split': 'train',
		'img_resize': 256,
		'data_root': 'images/',	# MODIFY PATH ACCORDINGLY
    		"tag_list": 'final_train.p',
		'data_list': 'development_kit/data/train.txt'	# MODIFY PATH ACCORDINGLY	
	}

	params_val = {
		'name': 'miniplaces2',
		'split': 'val',
		'img_resize': 256,
		'data_root': 'images/',	# MODIFY PATH ACCORDINGLY
    		'tags_file': 'final_val.p',
		'data_list': 'development_kit/data/val.txt'		# MODIFY PATH ACCORDINGLY
	}

	params_test = {
		'name': 'miniplaces2',
		'split': 'test',
		'img_resize': 256,
		'data_root': 'images/',	# MODIFY PATH ACCORDINGLY
    		'tag_list': 'final_test.p',
		'data_list': 'development_kit/data/test_new.txt'		# MODIFY PATH ACCORDINGLY
	}
	
	createH5(params_train)
	createH5(params_val)
	createH5(params_test)
