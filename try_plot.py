import matplotlib.pyplot as plt
import numpy as np
import os

script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'Results/')
sample_file_name = "try_plot"

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)


y = np.array([0,3,5,9,11])
# x = [0, 1, 2, 3, 4]
# y = [0, 3, 5, 9, 11]
plt.plot(y)
plt.xlabel('Months')
plt.ylabel('Books Read')
plt.savefig(results_dir + sample_file_name)
# plt.show()