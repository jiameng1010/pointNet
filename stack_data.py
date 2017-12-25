import h5py
import os
import numpy as np

which = 'testcomplete'
size = 1*2048 + 420

data = np.empty((size, 1024, 3))
label = np.empty((size, 1))
save_dir = './data/h5/'
for i in range(2):
    h5r = h5py.File((save_dir + which + str(i) + '.h5'), 'r')
    if not i == 1:
        data[(i*2048):((i+1)*2048), :, :] = h5r['data'][:]
        label[(i*2048):((i+1)*2048), :] = h5r['label'][:]
    else:
        data[(i*2048):, :, :] = h5r['data'][:]
        label[(i*2048):, :] = h5r['label'][:]
    h5r.close()

h5w = h5py.File((save_dir + which + "all" + '.h5'), 'w')
h5w.create_dataset('data', data=data)
h5w.create_dataset('label', data=label)