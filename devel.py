import numpy as np
import scipy

from main import Task

# example of testing a specific task
id = 5                      # the grid task id
running_id = 0              # the id indicating the i-th execution of the task
T = Task(id, running_id)    # initialize task
result = T.run()
print (result)