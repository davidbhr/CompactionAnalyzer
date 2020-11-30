# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 14:58:46 2020

@author: david
"""

import numpy as np
import matplotlib.pyplot as plt

plt.subplot(121)
# mean over angles
vec1 = np.array([0,1])
angle = np.random.uniform(0,2*np.pi, (10000)) 


cosinus = np.cos (angle)


print (np.mean(cosinus))
plt.hist(angle,bins=100)
plt.hist(cosinus,bins=100)
plt.show()