from DataLoader import *

# Dataset Parameters
batch_size = 80#200
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])


opt_data_test = {
    'data_h5': 'miniplaces_256_test.h5',
    'data_root': 'images/',   # MODIFY PATH ACCORDINGLY
    'data_list': 'development_kit/data/test.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

loader_test = DataLoaderH5(**opt_data_test)

images_batch, labels_batch = loader_test.next_batch(5) 

import scipy.misc
scipy.misc.imsave('outfile0.jpg', images_batch[0])

print labels_batch
