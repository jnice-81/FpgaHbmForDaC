#define NUM_32B 16
#define NUM_BURST 256

extern "C" {
    void comp_unit(const unsigned long long size, const float scale,
        const float* in, float* out) {
            #pragma HLS INLINE

            burst_loop:
        for(unsigned long long b = 0; b < size; b += NUM_BURST) {
            #pragma HLS pipeline II=16

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

    void scal(const unsigned long long size, const float scale,
                float *in0,
                float *in1,
                float *in2,
                float *in3,
                float *in4,
                float *in5,
                float *in6,
                float *in7,
                float *in8,
                float *in9,
                float *in10,
                float *in11,
                float *in12,
                float *in13,
                float *in14,
                float *in15,
                float *out0,
                float *out1,
                float *out2,
                float *out3,
                float *out4,
                float *out5,
                float *out6,
                float *out7,
                float *out8,
                float *out9,
                float *out10,
                float *out11,
                float *out12,
		        float *out13,
                float *out14,
                float *out15)
    {
        #pragma HLS INTERFACE m_axi port = in0 offset = slave bundle = gmemIn0
        #pragma HLS INTERFACE m_axi port = in1 offset = slave bundle = gmemIn1
        #pragma HLS INTERFACE m_axi port = in2 offset = slave bundle = gmemIn2
        #pragma HLS INTERFACE m_axi port = in3 offset = slave bundle = gmemIn3
        #pragma HLS INTERFACE m_axi port = in4 offset = slave bundle = gmemIn4
        #pragma HLS INTERFACE m_axi port = in5 offset = slave bundle = gmemIn5
        #pragma HLS INTERFACE m_axi port = in6 offset = slave bundle = gmemIn6
        #pragma HLS INTERFACE m_axi port = in7 offset = slave bundle = gmemIn7
        #pragma HLS INTERFACE m_axi port = in8 offset = slave bundle = gmemIn8
        #pragma HLS INTERFACE m_axi port = in9 offset = slave bundle = gmemIn9
        #pragma HLS INTERFACE m_axi port = in10 offset = slave bundle = gmemIn10
        #pragma HLS INTERFACE m_axi port = in11 offset = slave bundle = gmemIn11
        #pragma HLS INTERFACE m_axi port = in12 offset = slave bundle = gmemIn12
        #pragma HLS INTERFACE m_axi port = in13 offset = slave bundle = gmemIn13
        #pragma HLS INTERFACE m_axi port = in14 offset = slave bundle = gmemIn14
        #pragma HLS INTERFACE m_axi port = in15 offset = slave bundle = gmemIn15
        
        #pragma HLS INTERFACE m_axi port = out0 offset = slave bundle = gmemOut0
        #pragma HLS INTERFACE m_axi port = out1 offset = slave bundle = gmemOut1
        #pragma HLS INTERFACE m_axi port = out2 offset = slave bundle = gmemOut2
        #pragma HLS INTERFACE m_axi port = out3 offset = slave bundle = gmemOut3
        #pragma HLS INTERFACE m_axi port = out4 offset = slave bundle = gmemOut4
        #pragma HLS INTERFACE m_axi port = out5 offset = slave bundle = gmemOut5
        #pragma HLS INTERFACE m_axi port = out6 offset = slave bundle = gmemOut6
        #pragma HLS INTERFACE m_axi port = out7 offset = slave bundle = gmemOut7
        #pragma HLS INTERFACE m_axi port = out8 offset = slave bundle = gmemOut8
        #pragma HLS INTERFACE m_axi port = out9 offset = slave bundle = gmemOut9
        #pragma HLS INTERFACE m_axi port = out10 offset = slave bundle = gmemOut10
        #pragma HLS INTERFACE m_axi port = out11 offset = slave bundle = gmemOut11
        #pragma HLS INTERFACE m_axi port = out12 offset = slave bundle = gmemOut12
        #pragma HLS INTERFACE m_axi port = out13 offset = slave bundle = gmemOut13
        #pragma HLS INTERFACE m_axi port = out14 offset = slave bundle = gmemOut14
        #pragma HLS INTERFACE m_axi port = out15 offset = slave bundle = gmemOut15

        #pragma HLS DATAFLOW

        comp_unit(size, scale, in0, out0);
        comp_unit(size, scale, in1, out1);
        comp_unit(size, scale, in2, out2);
        comp_unit(size, scale, in3, out3);
        comp_unit(size, scale, in4, out4);
        comp_unit(size, scale, in5, out5);
        comp_unit(size, scale, in6, out6);
        comp_unit(size, scale, in7, out7);
        comp_unit(size, scale, in8, out8);
        comp_unit(size, scale, in9, out9);
        comp_unit(size, scale, in10, out10);
        comp_unit(size, scale, in11, out11);
        comp_unit(size, scale, in12, out12);
        comp_unit(size, scale, in13, out13);
        comp_unit(size, scale, in14, out14);
        comp_unit(size, scale, in15, out15);
    }
}
