#define UNROLL_FACTOR 16
#define NUM_BURST 256

extern "C" {
    void scal(const float *in, float *out, const unsigned int size, const float scale) {
        #pragma HLS INTERFACE m_axi port = in offset = slave bundle = gmemIn
        #pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmemOut
    
        for(unsigned int b = 0; b < size; b += NUM_BURST) {
            #pragma HLS pipeline II=16

            for(unsigned int i = 0; i < NUM_BURST; i += UNROLL_FACTOR) {
                for(unsigned int j = 0; j < UNROLL_FACTOR; j++) {
                    #pragma HLS unroll skip_exit_check
                    out[b+i+j] = scale * in[b+i+j];
                }
            }
        }
    }
}