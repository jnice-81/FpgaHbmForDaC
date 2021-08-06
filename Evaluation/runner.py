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

def run_and_time(run_program):
    run_program()

def run_axpy(input_size, banks_per_input, verify_only=True):
    x = rand_arr([input_size])
    y = rand_arr([input_size])
    z = rand_arr([input_size])
    if verify_only:
        expect = x + y

    sdfg = only_hbm_axpy_sdfg(banks_per_input)
    exec = lambda: sdfg(x=x, y=y, z=z, N=input_size)
    if verify_only:
        exec()
        assert np.allclose(z, expect)
    else:
        run_and_time(exec)

def run_gemv(m, n, banks_A, no_split_y, verify_only=True):
    A = rand_arr([m, n])
    x = rand_arr([n])
    y = rand_arr([m])
    if verify_only:
        expect = A @ x

    sdfg = only_hbm_gemv_sdfg(banks_A, no_split_y)
    exec = lambda: sdfg(A=A, x=x, y=y, M=m, N=n)
    if verify_only:
        exec()
        assert np.allclose(y, expect)
    else:
        run_and_time(exec)

def run_dot(input_size, banks_per_input, verify_only=True):
    x = rand_arr([input_size])
    y = rand_arr([input_size])
    result = rand_arr([1])
    if verify_only:
        expect = np.dot(x, y)
    sdfg = only_hbm_dot_sdfg(banks_per_input)
    exec = lambda: sdfg(x=x, y=y, final_result=result, N=input_size)
    if verify_only:
        exec()
        assert np.allclose(result, expect)
    else:
        run_and_time(exec)


def check_correct(size_control, num_banks, what="all", second_size=None):
    if second_size == None:
        second_size = size_control

    if what == "all" or what == "axpy":
        run_axpy(1024*num_banks*size_control, num_banks, True)
    if what == "all" or what == "gemv":
        run_gemv(1024*num_banks*size_control, 32*num_banks*second_size , 2, True)
    if what == "all" or what == "dot":
        run_dot(16*num_banks*size_control, 2, True)
