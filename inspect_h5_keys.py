import h5py; f = h5py.File('boredom_hdf5/S12_Boredom.h5', 'r'); obj=f[list(f.keys())[0]]; print(obj.keys())
