import h5py; f = h5py.File('boredom_hdf5/S12_Boredom.h5', 'r'); k=list(f.keys())[0]; d=f[k]['eeg']; print(d.attrs['chOrder'] if 'chOrder' in d.attrs else 'Missing')
