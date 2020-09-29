# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:16:51 2020

@author: david
"""
import numpy as np
import clickpoints
import matplotlib.pyplot as plt

# load database
db = clickpoints.DataFile("pos.cdb")
im_entry = db.getImage(0)
im_pixel = im_entry.data


# get the markers in the image
markers = np.array(db.getMarkers(image=im_entry, type="cells"))
# save positions
np.save("markers.npy", markers)

#show data
plt.figure()
plt.axis("off")
plt.imshow(im_pixel, cmap="gray")
plt.plot(markers[:, 0], markers[:, 1], 'C0o', ms=2)
plt.savefig("clicks.png", dpi=400, bbox_inches="tight")




