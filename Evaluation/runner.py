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

def run_gemv(m, n, banks_A, no_split_y):
    A = rand_arr([m, n])
    x = rand_arr([n])
    y = rand_arr([m])
    expect = A @ x

    sdfg = only_hbm_gemv_sdfg(banks_A, no_split_y)
    run_and_time(sdfg, A=A, x=x, y=y)
    assert np.allclose(y, expect)

run_gemv(1024*2, 32*2, 2, False)