from dace.codegen.targets import fpga
from dace import dtypes
from hbm_axpy_dot_based import *
from hbm_gemv_based import *
from hbm_ger import *
import numpy as np
import argparse
import pandas as pd
from dace.config import Config

def rand_arr(shape):
    fix = True

    if fix:
        a = np.ones(shape) * 1
    else:
        a = np.random.rand(*shape)
    a = a.astype(np.float32)
    return a


measure_time = False
measure_write_N = 0
measure_append_to_file = None

def run_and_time(sdfg: SDFG, **kwargs):
    if measure_append_to_file is None:
        repeat_timing = 1
    else:
        repeat_timing = 30

    if measure_time:
        if Config.get("compiler", "xilinx", "mode") != "hardware":
            raise RuntimeError("time can only be measured with a hardware kernel") # Protect from accidentally overwritting with emulation
        for state in sdfg.states():
            if fpga.is_fpga_kernel(sdfg, state):
                state.instrument = dtypes.InstrumentationType.FPGA
            else:
                pass
                #state.instrument = dtypes.InstrumentationType.Timer
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
                        print(name)
                        total_time += time[0]
                    if "Full FPGA kernel" in name:
                        kernel_time = time[0]
            times.append([f"{sdfg.name}_{measure_write_N}", measure_write_N, total_time, kernel_time])
        if measure_append_to_file is not None:
            report = pd.DataFrame(columns=["name", "N", "total_time", "kernel_time"], data=times)
            report.to_csv("times.csv", index=False, mode='a', header=False)
        else:
            data_mul_factor = 2 * 4
            print(f"Assuming IO size = measureWriteN*{data_mul_factor} (measure_write_N={measure_write_N}):")
            print(f"IO Speed: {(measure_write_N*data_mul_factor)/ (times[0][3])}")
    else:
        executable = sdfg.compile()
        executable(**kwargs)
    

def run_axpy(input_size, banks_per_input, verify=True):
    x = rand_arr([input_size])
    y = rand_arr([input_size])
    z = rand_arr([input_size])
    a = rand_arr([1])
    if verify:
        expect = x + a * y

    sdfg = only_hbm_axpy_sdfg(banks_per_input)
    run_and_time(sdfg, x=x, y=y, z=z, N=input_size, alpha=a[0])
    if verify:
        assert np.allclose(z, expect)
        print("Verified")

def run_dot(input_size, banks_per_input, verify=True):
    x = rand_arr([input_size])
    y = rand_arr([input_size])
    result = rand_arr([1])
    if verify:
        expect = np.dot(x, y)
    sdfg = only_hbm_dot_sdfg(banks_per_input)
    run_and_time(sdfg, x=x, y=y, final_result=result, N=input_size)
    if verify:
        assert np.allclose(result, expect)
        print("Verified")

def run_axpydot(input_size, banks_per_input, verify=True):
    x = rand_arr([input_size])
    y = rand_arr([input_size])
    w = rand_arr([input_size])
    a = rand_arr([1])
    result = rand_arr([1])
    if verify:
        expect = np.dot(x+ a[0] * y, w)
    
    sdfg = hbm_axpy_dot(banks_per_input)
    run_and_time(sdfg, axpy_x=x, axpy_y=y, dot_y=w, final_result=result, N=input_size, alpha=a[0])
    if verify:
        assert np.allclose(result[0], expect)
        print("Verified")

def run_gemv(m, n, banks_A, verify=True):
    A = rand_arr([m, n])
    x = rand_arr([n])
    y = rand_arr([m])
    if verify:
        expect = A @ x

    sdfg = only_hbm_gemv_sdfg(banks_A)
    run_and_time(sdfg, A=A, x=x, y=y, M=m, N=n)
    if verify:
        assert np.allclose(y, expect)
        print("Verified")

def run_ger(m, n, banks_A, verify=True):
    A = rand_arr([m, n])
    x = rand_arr([m])
    y = rand_arr([n])
    res = rand_arr([m, n])
    alpha = rand_arr([1])
    if verify:
        expect = rand_arr([m, n])
        for i in range(m):
            expect[i, :] = A[i, :] + alpha[0] * x[i] * y

    sdfg = hbm_ger_sdfg(banks_A, 1024, 1)
    run_and_time(sdfg, A=A, x=x, y=y, res=res, alpha=alpha[0], m=m, n=n)
    if verify:
        assert np.allclose(res, expect)
        print("Verified")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("app", type=str, help="Applications are axpy, dot, gemv, axpydot.")
    parser.add_argument("size", type=int, help="A value controlling the size of the input data")
    parser.add_argument("--show", type=bool, help="If True show html-view of the sdfg. If false runs the sdfg")
    parser.add_argument("--time", type=bool, help="Measure the execution time. Pass True explicit.")
    parser.add_argument("--reportfile", type=str, help="Append measured times to the proved csv file. Only has an influence when --time is true.")
    args = parser.parse_args()

    # multiplications appended at the end don't have a functional meaning and are just added to scale to a reasonable value
    if args.app == "axpy":
        num_banks = 10
        input_size = 16*4096*num_banks*args.size
    elif args.app == "dot":
        num_banks = 15 # DDR 0 has a maximum of 15 attached interfaces on u280
        input_size = 8*8192*num_banks*args.size
    elif args.app == "gemv":
        num_banks = 24
        m = 32*num_banks*args.size  * 11
        n = 1024*8*args.size
        input_size = m*n
        print(f"INPUT SIZE: {m}x{n}")
    elif args.app == "axpydot":
        num_banks = 10
        input_size = 8*8192*num_banks*args.size
    elif args.app == "ger":
        num_banks = 12
        m = 1*num_banks*args.size * 683
        n = 1024*8*args.size
        input_size = m*n
        print(f"INPUT SIZE: {m}x{n}")

    measure_time = args.time
    measure_write_N = input_size // (1000*1000)
    measure_append_to_file = args.reportfile

    if args.app == "axpy":
        if args.show:
            sdfg = only_hbm_axpy_sdfg(num_banks)
            sdfg.view()
        else:
            run_axpy(input_size, num_banks, not args.time)
    if args.app == "dot":
        if args.show:
            sdfg = only_hbm_dot_sdfg(num_banks)
            sdfg.view()
        else:
            run_dot(input_size, num_banks, not args.time)
    if args.app == "axpydot":
        if args.show:
            sdfg = hbm_axpy_dot(num_banks)
            sdfg.view()
        else:
            run_axpydot(input_size, num_banks, not args.time)
    if args.app == "gemv":
        if args.show:
            sdfg = only_hbm_gemv_sdfg(num_banks)
            sdfg.view()
        else:
            run_gemv(m, n, num_banks, not args.time)
    if args.app == "ger":
        if args.show:
            sdfg = hbm_ger_sdfg(num_banks, 1024, 1)
            sdfg.view()
        else:
            run_ger(m, n, num_banks, not args.time)
