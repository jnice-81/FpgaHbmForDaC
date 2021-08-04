#define NUM_32B 16
#define NUM_BURST 64

struct floatarrayarg
{
    float infloats[NUM_32B];
};

extern "C" {
    void scal(floatarrayarg *in, floatarrayarg *out, const unsigned long long size, const float scale) {
        //#pragma HLS INTERFACE m_axi port = in offset = slave bundle = gmem max_read_burst_length = 64
        //#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem max_read_burst_length = 64

        #pragma HLS INTERFACE m_axi port = in offset = slave bundle = gmem
        #pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem

        #pragma HLS INTERFACE s_axilite port=in
        #pragma HLS INTERFACE s_axilite port=out
        #pragma HLS INTERFACE s_axilite port=return
        #pragma HLS INTERFACE s_axilite port=size
        #pragma HLS INTERFACE s_axilite port=scale

        unsigned long long customsize = size / NUM_32B; //Since NUM_32B == Number of floats in floatarrayarg

        burst_loop:
        for(unsigned long long b = 0; b < customsize; b += NUM_BURST) {
            #pragma HLS pipeline rewind II=64

            pipeline_loop: 
            for(unsigned long long i = 0; i < NUM_BURST; i += 1) {

                parallel_loop:
                for(unsigned int j = 0; j < NUM_32B; j++) {
                    #pragma HLS unroll skip_exit_check
                    #pragma depencence variable=out intra false
                    #pragma dependence variable=in intra false

                    out[b+i].infloats[j] = scale * in[b+i].infloats[j];
                }
            }
        }
    }
}
