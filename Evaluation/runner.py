from dace.codegen.targets import fpga
from dace import dtypes
from hbm_blas_operators import *
import numpy as np
import argparse
import pandas as pd
from dace.config import Config

def rand_arr(shape):
    fix = False

    if fix:
        a = np.ones(shape)
    else:
        a = np.random.rand(*shape)
    a = a.astype(np.float32)
    return a

def run_and_time(sdfg: SDFG, measure_time=False, **kwargs):
    repeat_timing = 1

    if measure_time:
        if Config.get("compiler", "xilinx", "mode") != "hardware":
            raise RuntimeError("time can only be measured with a hardware kernel") # Protect from accidentally overwritting with emulation
        for state in sdfg.states():
            if fpga.is_fpga_kernel(sdfg, state):
                state.instrument = dtypes.InstrumentationType.FPGA
            else:
                state.instrument = dtypes.InstrumentationType.Timer
        executable = sdfg.compile()
        times = []
        for i in range(repeat_timing):
            executable(**kwargs)
            report = sdfg.get_latest_report()
            print(report)
            total_time = 0
            for duration in report.durations.values():
                for name, time in duration.items():
                    if "pre" in name or "post" in name or "Full FPGA state" in name:
                        total_time += time[0]
                    elif "Full FPGA kernel" in name:
                        kernel_time = time[0]
            times.append([total_time, kernel_time])
        report = pd.DataFrame(columns=["total_time", "kernel_time"], data=times)
        report.to_csv(f"time_reports/{sdfg.name}_times", index=False)
    else:
        executable = sdfg.compile()
        executable(**kwargs)
    

def run_axpy(input_size, banks_per_input, verify=True, measure_time=False):
    x = rand_arr([input_size])
    y = rand_arr([input_size])
    z = rand_arr([input_size])
    if verify:
        expect = x + y

    sdfg = only_hbm_axpy_sdfg(banks_per_input)
    run_and_time(sdfg, measure_time, x=x, y=y, z=z, N=input_size)
    if verify:
        assert np.allclose(z, expect)
        print("Verified")

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
    result = rand_arr([1])
    if verify_only:
        expect = np.dot(x+y, w)
    
    sdfg = hbm_axpy_dot(banks_per_input)
    exec = lambda: sdfg(axpy_x=x, axpy_y=y, dot_y=w, final_result=result, N=input_size)
    if verify_only:
        exec()
        assert np.allclose(result[0], expect)
    else:
        run_and_time(exec)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("app", type=str, help="Applications are axpy, dot, gemv, axpydot.")
    parser.add_argument("size", type=int, help="A value controlling the size of the input data")
    parser.add_argument("--show", type=bool, help="If True show html-view of the sdfg. If false runs the sdfg")
    parser.add_argument("--time", type=bool, help="If True show html-view of the sdfg. If false runs the sdfg")
    parser.add_argument("--secondsize", type=int, default=1, help="A second value controlling the size of input data")
    args = parser.parse_args()

    if args.app == "axpy":
        num_banks = 10
    elif args.app == "dot":
        num_banks = 15 # DDR 0 has a maximum of 15 attached interfaces on u280
    elif args.app == "gemv":
        num_banks = 30
    elif args.app == "axpydot":
        num_banks = 10

    if args.secondsize == None:
        second_size = args.size

    if args.app == "axpy":
        if args.show:
            sdfg = only_hbm_axpy_sdfg(num_banks)
            sdfg.view()
        else:
            run_axpy(16*64*num_banks*args.size, num_banks, not args.time, args.time)
    if args.app == "gemv":
        raise NotImplementedError()
        if args.show:
            sdfg = only_hbm_gemv_sdfg(num_banks, True)
            sdfg.view()
        else:
            run_gemv(1024*num_banks*args.size, 32*num_banks*second_size , num_banks, not args.time, args.time)
    if args.app == "dot":
        if args.show:
            sdfg = only_hbm_dot_sdfg(num_banks)
            sdfg.view()
        else:
            run_dot(8*64*num_banks*args.size, num_banks, True)
    if args.app == "axpydot":
        if args.show:
            sdfg = hbm_axpy_dot(num_banks)
            sdfg.view()
        else:
            run_axpydot(8*64*num_banks*args.size, num_banks, True)