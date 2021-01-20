# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:02:06 2020

@author: david
"""
import numpy as np
import matplotlib.pyplot as plt

# x = np.arange(0,2*np.pi,0.1)

# np.cos(x)

# plt.xticks( x[::5],[np.round(i*180/np.pi) for i in x[::5]])
# plt.plot(x, np.cos(x))



from PIL import Image
for t in range(10):
    i0 = (Image.open(r"D:\david\Dropbox\software-github\CompactionAnalyzer\Tutorial\EmptyGel\0\RT+37C_pore_size_B03_G002_0001_0001_C003Z045.tif"))  #ode  #1024 1024
    
    im = np.random.uniform(0,np.max(i0), (1024,1024)).astype(np.uint16)
    
    Image.fromarray(im).save(fr"D:\david\Dropbox\software-github\CompactionAnalyzer\Tutorial\EmptyGel\0\rand{t}.tif")
