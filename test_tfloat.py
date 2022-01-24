import numpy.core.tracked_float as tf
import numpy as np

import time
#print(dir(tf))
x = np.ones((1, 10)).astype(tf.tracked_float)
tf.initialize(x, 1)
np.save('/test2', x)
# print(z[0, 0].pnum)
# print(z[0, 0].num)
# print(z[0,0].num)
# print(z[0,0].num)
# print(z[0,0].num)
# print(z[0,0].num)
# print(z[0,0].num)
# print(z[0,0].num)
# print(z[0,0].num)
# print(z[0,0].num)
# print(z[0,0].num)
# z = np.sum(y, axis = 1, dtype=tf.tracked_float, initial=None)

# print(z[1].n)
# print(z[1].n)