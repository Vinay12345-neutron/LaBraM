import h5py; f = h5py.File('boredom_hdf5/S12_Boredom.h5', 'r'); print(f.keys()); k=list(f.keys())[0]; print(type(f[k])); print(f[k].keys() if hasattr(f[k], 'keys') else 'Not a group')
