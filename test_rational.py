from numpy.core._rational_tests import rational
import numpy as np

r = np.random.randint(10, size =(10, 10)).astype(rational)
np.save('./test.np', r)