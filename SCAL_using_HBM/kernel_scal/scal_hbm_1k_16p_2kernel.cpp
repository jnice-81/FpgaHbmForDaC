#define NUM_32B 16 
#define NUM_BURST 256

extern "C" {

    //Expects size = number of elements in each of the 8 accessed memories (not the sum)

    void scal(const unsigned long long size, const float scale,
        const float* in0,
        const float* in1,
        const float *in2,
        const float *in3,
        const float *in4,
        const float *in5,
        const float *in6,
        const float *in7,
        float* out0,
        float* out1,
        float* out2,
        float* out3,
        float* out4,
        float* out5,
        float* out6,
        float* out7) {

        #pragma HLS INTERFACE m_axi port = in0 offset = slave bundle = gmemIn0 
        #pragma HLS INTERFACE m_axi port = in1 offset = slave bundle = gmemIn1
        #pragma HLS INTERFACE m_axi port = in2 offset = slave bundle = gmemIn2
        #pragma HLS INTERFACE m_axi port = in3 offset = slave bundle = gmemIn3
        #pragma HLS INTERFACE m_axi port = in4 offset = slave bundle = gmemIn4
        #pragma HLS INTERFACE m_axi port = in5 offset = slave bundle = gmemIn5
        #pragma HLS INTERFACE m_axi port = in6 offset = slave bundle = gmemIn6
        #pragma HLS INTERFACE m_axi port = in7 offset = slave bundle = gmemIn7

        #pragma HLS INTERFACE m_axi port = out0 offset = slave bundle = gmemOut0
        #pragma HLS INTERFACE m_axi port = out1 offset = slave bundle = gmemOut1
        #pragma HLS INTERFACE m_axi port = out2 offset = slave bundle = gmemOut2
        #pragma HLS INTERFACE m_axi port = out3 offset = slave bundle = gmemOut3
        #pragma HLS INTERFACE m_axi port = out4 offset = slave bundle = gmemOut4
        #pragma HLS INTERFACE m_axi port = out5 offset = slave bundle = gmemOut5
        #pragma HLS INTERFACE m_axi port = out6 offset = slave bundle = gmemOut6
        #pragma HLS INTERFACE m_axi port = out7 offset = slave bundle = gmemOut7

        for(unsigned long long b = 0; b < size; b += NUM_BURST) {
            #pragma HLS pipeline II=16

            for(unsigned long long i = 0; i < NUM_BURST; i += NUM_32B) {

                for(unsigned int j = 0; j < NUM_32B; j++) {
                    #pragma HLS unroll skip_exit_check

                    out0[b+i+j] = scale * in0[b+i+j];
                    out1[b+i+j] = scale * in1[b+i+j];
                    out2[b+i+j] = scale * in2[b+i+j];
                    out3[b+i+j] = scale * in3[b+i+j];
                    out4[b+i+j] = scale * in4[b+i+j];
                    out5[b+i+j] = scale * in5[b+i+j];
                    out6[b+i+j] = scale * in6[b+i+j];
                    out7[b+i+j] = scale * in7[b+i+j];
                }
            }
        }
    }
}
