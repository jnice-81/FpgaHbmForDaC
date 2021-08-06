from hbm_blas_operators import *
import numpy as np

def rand_arr(shape):
    fix = False

    if fix:
        a = np.ones(shape)
    else:
        a = np.random.rand(*shape)
    a = a.astype(np.float32)
    return a

def run_and_time(sdfg, **kwargs):
    sdfg(**kwargs)

def run_axpy(input_size, banks_per_input):
    x = rand_arr([input_size])
    y = rand_arr([input_size])
    z = rand_arr([input_size])
    expect = x + y

    sdfg = only_hbm_axpy_sdfg(banks_per_input)
    run_and_time(sdfg, x=x, y=y, z=z)
    assert np.allclose(z, expect)

def run_dot(input_size, banks_per_input):
    x = rand_arr([input_size])
    y = rand_arr([input_size])
    result = rand_arr([1])
    expect = np.dot(x, y)

    sdfg = only_hbm_dot_sdfg(banks_per_input)
    run_and_time(sdfg, x=x, y=y, final_result=result)
    assert np.allclose(result, expect)

run_dot(1024*2, 2)