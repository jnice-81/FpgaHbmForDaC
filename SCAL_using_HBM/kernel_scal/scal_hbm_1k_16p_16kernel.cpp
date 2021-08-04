#define NUM_32B 16
#define NUM_BURST 256

extern "C" {
    void scal(float *in, float *out, const unsigned long long size, const float scale) {
        /*
            For future reference: Each m_axi port has a seperate read and write port, so
            one can be used for one read arg and one write arg without lost performance.
            If several read/writes are needed consider adding more via bundle argument.

            The scalar variables get mapped to s_axelite ports.
        */
       
       //Two interfaces are used here, since otherwise bursting is only possible in one direction (see VITIS HLS Optimization Burst access)!!! --This statement might actually be a mistake in the docs
        //#pragma HLS INTERFACE m_axi port = in offset = slave bundle = gmemIn max_read_burst_length = 256 num_read_outstanding=12 num_write_outstanding=1 max_write_burst_length=2
        //#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmemOut max_write_burst_length = 256 num_write_outstanding=12 num_read_outstanding=1 max_read_burst_length=2

        #pragma HLS INTERFACE m_axi port = in offset = slave bundle=gmem
        #pragma HLS INTERFACE m_axi port = out offset = slave bundle=gmem

        #pragma HLS INTERFACE s_axilite port=in
        #pragma HLS INTERFACE s_axilite port=out
        #pragma HLS INTERFACE s_axilite port=return
        #pragma HLS INTERFACE s_axilite port=size
        #pragma HLS INTERFACE s_axilite port=scale

        burst_loop:
        for(unsigned long long b = 0; b < size; b += NUM_BURST) {
            #pragma HLS pipeline II=16

            //reads/writes 1 KB of data during one burst
            pipeline_loop: 
            for(unsigned long long i = 0; i < NUM_BURST; i += NUM_32B) {

                parallel_loop:
                for(unsigned int j = 0; j < NUM_32B; j++) {
                    #pragma HLS unroll skip_exit_check
                    out[b+i+j] = scale * in[b+i+j];
                }
            }
        }
    }
}
