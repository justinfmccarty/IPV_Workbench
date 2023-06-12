
import multiprocessing
from multiprocessing.managers import SharedMemoryManager
from multiprocessing import shared_memory
from multiprocessing.shared_memory import SharedMemory
from typing import Tuple
import numpy as np

# def create_shared_memory_nparray(data, data_name):
#     shared_memory.SharedMemory(create=True, size=data.size * data.itemsize, name=data_name)
#
#
# def reopen_memory(array_shape, shm_name, dtype_=np.float32):
#     existing_shm = shared_memory.SharedMemory(name=shm_name)
#     return np.ndarray(array_shape, dtype=dtype_, buffer=existing_shm.buf)


def create_read_only_arrays(original_data, data_name):
    print(data_name)
    shm = shared_memory.SharedMemory(create=True, size=original_data.nbytes, name=data_name)
    new_array = np.ndarray(original_data.shape, dtype=original_data.dtype, buffer=shm.buf)
    np.copyto(new_array, original_data)


def access_array(shared_name, original_shape, original_dtype):
    shm = shared_memory.SharedMemory(name=shared_name)
    arr = np.ndarray(original_shape, dtype=original_dtype, buffer=shm.buf)
    return arr
    # return np.frombuffer(buffer=shm.buf, dtype=original_dtype).reshape(original_shape)


def release_shared(name):
    shm = shared_memory.SharedMemory(name=name)
    shm.close()
    shm.unlink()

# def create_np_array_from_shared_mem(shared_mem, shared_data_dtype, shared_data_shape):
#     arr = np.frombuffer(shared_mem.buf, dtype=shared_data_dtype)
#     arr = arr.reshape(shared_data_shape)
#     return arr

def create_np_array_from_shared_mem(
        shared_mem: SharedMemory, shared_data_dtype: np.dtype, shared_data_shape: Tuple[int, ...]
) -> np.ndarray:
    arr = np.frombuffer(shared_mem.buf, dtype=shared_data_dtype)
    arr = arr.reshape(shared_data_shape)
    return arr

def unpack_shared_tuple(shared_tup):
    mem, dtype, shape = shared_tup
    return create_np_array_from_shared_mem(mem, dtype, shape)

def test_module():
    print("world.")