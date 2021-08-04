#define NUM_32B 16

extern "C" {
    void scal(float *in, float *out, const unsigned long long size, const float scale) {
        #pragma HLS INTERFACE m_axi port = in offset = slave bundle=gmem
        #pragma HLS INTERFACE m_axi port = out offset = slave bundle=gmem

        #pragma HLS INTERFACE s_axilite port=in
        #pragma HLS INTERFACE s_axilite port=out
        #pragma HLS INTERFACE s_axilite port=return
        #pragma HLS INTERFACE s_axilite port=size
        #pragma HLS INTERFACE s_axilite port=scale

        pipeline_loop: 
        for(unsigned long long i = 0; i < size; i += NUM_32B) {

            parallel_loop:
            for(unsigned int j = 0; j < NUM_32B; j++) {
                #pragma HLS unroll skip_exit_check
                out[i+j] = scale * in[i+j];
            }
        }
    }
}
