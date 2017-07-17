
import h5py

weights_file = 'vae_weights.h5'

f = h5py.File(weights_file)

for x in f.attrs.items():
    print(x)


layer_names = f.attrs['layer_names']
for name in layer_names:
    print(name)

#%%

for x in f.items():
    print(x)
    
#%%   
g = f['decoder']
for x in g.items():
    print(x)
    

h = f['/decoder/deconv_1']
for x in h.items():
    print(x)
    
f.close()