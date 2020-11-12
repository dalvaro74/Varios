import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import data.data01 as d01

cv_error = [round(num,3) for num in d01.cv_error]
cv_std = [round(num,3) for num in d01.cv_std]

plt.plot(range(1,len(cv_error)+1),cv_error,'-o')
plt.errorbar(range(1,len(cv_error)+1), cv_error, yerr=cv_std, fmt='o')
plt.xlabel('# features')
plt.ylabel('CV error')
plt.show()

