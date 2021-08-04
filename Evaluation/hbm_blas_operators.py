import dace
from hbm_transform import HbmTransform
from hbm_bank_split import HbmBankSplit
from hbm_transform import set_shape
from dace.transformation.optimizer import Optimizer

def axpy_sdfg(vec_size: int, banks_per_input: int):
    N = dace.symbol("N")

    @dace.program
    def axpy(x: dace.vector(dace.float32, vec_size)[N / vec_size], y: dace.vector(dace.float32, vec_size)[N / vec_size]):
        for i in dace.map[0:N/vec_size]:
            with dace.tasklet:
                xin << x[i]
                yin << y[i]
                yout >> y[i]
                yout = xin + yin
    sdfg = axpy.to_sdfg()
    sdfg.apply_strict_transformations()
    sdfg.arrays["x"].location["memorytype"] = "HBM"
    sdfg.arrays["x"].location["bank"] = f"0:{banks_per_input}"
    sdfg.apply_transformations(HbmTransform)
    sdfg.apply_strict_transformations()
    sdfg.apply_fpga_transformations()
    set_shape(sdfg.arrays["x"], [N/vec_size])
    set_shape(sdfg.arrays["y"], [N/vec_size])
    for match in Optimizer(sdfg).get_pattern_matches(patterns=HbmBankSplit):
        match.apply(sdfg)
    return sdfg

