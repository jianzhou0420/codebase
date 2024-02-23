
import numpy as np
import matplotlib.pyplot as plt
img=np.load('save.npy')

img=img/img.max()

plt.imshow(img)
plt.show()
