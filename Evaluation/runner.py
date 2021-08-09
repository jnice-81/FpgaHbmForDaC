from hbm_blas_operators import *
import numpy as np
import argparse

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

def run_axpydot(input_size, banks_per_input, verify_only=True):
    x = rand_arr([input_size])
    y = rand_arr([input_size])
    w = rand_arr([input_size])
    result = rand_arr([banks_per_input])
    if verify_only:
        expect = np.dot(x+y, w)
    
    sdfg = hbm_axpy_dot(banks_per_input)
    exec = lambda: sdfg(axpy_x=x, axpy_y=y, dot_y=w, result=result, N=input_size)
    if verify_only:
        exec()
        assert np.allclose(result.sum(), expect)
    else:
        run_and_time(exec)

def check_correct(size_control, num_banks, what, show_only=False, second_size=None):
    if second_size == None:
        second_size = size_control

    if what == "axpy":
        if show_only:
            sdfg = only_hbm_axpy_sdfg(num_banks)
            sdfg.view()
        else:
            run_axpy(1200*num_banks*size_control, num_banks, True)
    if what == "gemv":
        if show_only:
            sdfg = only_hbm_gemv_sdfg(num_banks, True)
            sdfg.view()
        else:
            run_gemv(1024*num_banks*size_control, 32*num_banks*second_size , num_banks, True)
    if what == "dot":
        if show_only:
            sdfg = only_hbm_dot_sdfg(num_banks)
            sdfg.view()
        else:
            run_dot(1200*num_banks*size_control, num_banks, True)
    if what == "axpydot":
        if show_only:
            sdfg = hbm_axpy_dot(num_banks)
            sdfg.view()
        else:
            run_axpydot(1200*num_banks*size_control, num_banks, True)

if __name__ == "__main__":
    check_correct(1, 2, "axpydot", False)
    exit()

    parser = argparse.ArgumentParser()
    parser.add_argument("app", type=str, help="Applications are axpy, dot, gemv, axpydot.")
    parser.add_argument("size", type=int, help="A value controlling the size of the input data")
    parser.add_argument("--show", type=bool, default=False, help="If True show html-view of the sdfg. If false runs the sdfg")
    parser.add_argument("--secondsize", type=int, default=1, help="A second valued controlling the size of input data")
    args = parser.parse_args()

    if args.app == "axpy":
        num_banks = 10
    elif args.app == "dot":
        num_banks = 15 # DDR 0 has a maximum of 15 attached interfaces on u280
    elif args.app == "gemv":
        num_banks = 30
    elif args.app == "axpydot":
        raise NotImplementedError
    check_correct(args.size, num_banks, args.app, args.show, args.secondsize)
