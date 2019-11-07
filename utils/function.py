import time
import os
# Function used for time record for different experiment
def time2str():
    time_id = str(int(time.time()))
    return time_id

def build_dirs(path):
    try:
        os.makedirs(path)
    except OSError as e:
        print(e)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False