import tensorflow as tf
from tensorflow.python.client import device_lib

def get_all_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

all_devices = get_all_devices()
print("all_devices: %s" % all_devices)
