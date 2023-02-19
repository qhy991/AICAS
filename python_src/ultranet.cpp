#include <stdint.h>
#define AP_INT_MAX_W 4096
#include <ap_int.h>
#include <hls_video.h>
#include "stream_tools.h"
#include "function.h"
#include "sliding_window_unit.h"
#include "matrix_vector_unit.h"
#include "config.h"
#include "param.h"
#include "conv2d.h"
#include "pool2d.h"

#define IN_IMAGE_WIDTH 32
#define IN_IMAGE_HEIGHT 32

void do_compute(stream<my_ap_axis> & in, stream<my_ap_axis> & out, const unsigned int reps) {
#pragma HLS DATAFLOW
    const unsigned int num_per_rep = 32 * 32 * 3 * 8 / 64;

    hls::stream<ap_uint<64> > in_stream_extract("in_stream_extract");
#pragma HLS STREAM variable=in_stream_extract depth=2 dim=1
    ExtractPixels<64, num_per_rep> (in, in_stream_extract, reps);

    hls::stream<ap_uint<64 * 3> > in_stream0("in_stream0");
#pragma HLS STREAM variable=in_stream0 depth=2 dim=1
    StreamingDataWidthConverter_Batch<64, 64 * 3, num_per_rep>(in_stream_extract, in_stream0, reps);

    hls::stream<ap_uint<CONV_0_0_IN_BIT * CONV_0_0_IFM_CH> > in_stream1_out("in_stream1_out");
#pragma HLS STREAM variable=in_stream1_out depth=2 dim=1
StreamingDataWidthConverter_Batch<64 * 3, CONV_0_0_IN_BIT * CONV_0_0_IFM_CH, num_per_rep / 3> (in_stream0, in_stream1_out, reps);

    // conv_0_0
    hls::stream<ap_uint<CONV_0_0_OUT_BIT * CONV_0_0_OFM_CH> > conv_0_0_out("conv_0_0_out");
#pragma HLS STREAM variable=conv_0_0_out depth=32 dim=1

    conv3x3_bn_act<
            CONV_0_0_K,
            CONV_0_0_S,
            CONV_0_0_P,
            CONV_0_0_IFM_ROW,
            CONV_0_0_IFM_COL,
            CONV_0_0_IFM_CH,
            CONV_0_0_IN_BIT,
            CONV_0_0_OFM_CH,
            CONV_0_0_OUT_BIT,
            CONV_0_0_W_BIT,
            32,
            CONV_0_0_BIAS_BIT,
            CONV_0_0_SIMD,
            CONV_0_0_PE,
            CONV_0_0_R_SHIFT>(
        in_stream1_out,
        conv_0_0_w,
        conv_0_0_bias,
        conv_0_0_out,
        reps);

    // conv_0_1
    hls::stream<ap_uint<CONV_0_1_OUT_BIT * CONV_0_1_OFM_CH> > conv_0_1_out("conv_0_1_out");
#pragma HLS STREAM variable=conv_0_1_out depth=32 dim=1

    conv3x3_bn_act<
            CONV_0_1_K,
            CONV_0_1_S,
            CONV_0_1_P,
            CONV_0_1_IFM_ROW,
            CONV_0_1_IFM_COL,
            CONV_0_1_IFM_CH,
            CONV_0_1_IN_BIT,
            CONV_0_1_OFM_CH,
            CONV_0_1_OUT_BIT,
            CONV_0_1_W_BIT,
            32,
            CONV_0_1_BIAS_BIT,
            CONV_0_1_SIMD,
            CONV_0_1_PE,
            CONV_0_1_R_SHIFT>(
        conv_0_0_out,
        conv_0_1_w,
        conv_0_1_bias,
        conv_0_1_out,
        reps);

    // maxpooling
    hls::stream<ap_uint<CONV_0_1_OUT_BIT * CONV_0_1_OFM_CH> > maxpool_0_out("maxpool_0_out");
#pragma HLS STREAM variable=maxpool_0_out depth=32 dim=1
    max_pool2d< 2,
            CONV_0_1_OFM_ROW,
            CONV_0_1_OFM_COL,
            CONV_0_1_OFM_CH,
            CONV_0_1_OUT_BIT>(
        conv_0_1_out,
        maxpool_0_out,
        reps);

    // conv_1_0
    hls::stream<ap_uint<CONV_1_0_OUT_BIT * CONV_1_0_OFM_CH> > conv_1_0_out("conv_1_0_out");
#pragma HLS STREAM variable=conv_1_0_out depth=32 dim=1

    conv3x3_bn_act<
            CONV_1_0_K,
            CONV_1_0_S,
            CONV_1_0_P,
            CONV_1_0_IFM_ROW,
            CONV_1_0_IFM_COL,
            CONV_1_0_IFM_CH,
            CONV_1_0_IN_BIT,
            CONV_1_0_OFM_CH,
            CONV_1_0_OUT_BIT,
            CONV_1_0_W_BIT,
            32,
            CONV_1_0_BIAS_BIT,
            CONV_1_0_SIMD,
            CONV_1_0_PE,
            CONV_1_0_R_SHIFT>(
        maxpool_0_out,
        conv_1_0_w,
        conv_1_0_bias,
        conv_1_0_out,
        reps);

    // conv_2_0
    hls::stream<ap_uint<CONV_2_0_OUT_BIT * CONV_2_0_OFM_CH> > conv_2_0_out("conv_2_0_out");
#pragma HLS STREAM variable=conv_2_0_out depth=32 dim=1

    conv3x3_bn_act<
            CONV_2_0_K,
            CONV_2_0_S,
            CONV_2_0_P,
            CONV_2_0_IFM_ROW,
            CONV_2_0_IFM_COL,
            CONV_2_0_IFM_CH,
            CONV_2_0_IN_BIT,
            CONV_2_0_OFM_CH,
            CONV_2_0_OUT_BIT,
            CONV_2_0_W_BIT,
            32,
            CONV_2_0_BIAS_BIT,
            CONV_2_0_SIMD,
            CONV_2_0_PE,
            CONV_2_0_R_SHIFT>(
        conv_1_0_out,
        conv_2_0_w,
        conv_2_0_bias,
        conv_2_0_out,
        reps);

    // conv_3_0
    hls::stream<ap_uint<CONV_3_0_OUT_BIT * CONV_3_0_OFM_CH> > conv_3_0_out("conv_3_0_out");
#pragma HLS STREAM variable=conv_3_0_out depth=32 dim=1

    conv3x3_bn_act<
            CONV_3_0_K,
            CONV_3_0_S,
            CONV_3_0_P,
            CONV_3_0_IFM_ROW,
            CONV_3_0_IFM_COL,
            CONV_3_0_IFM_CH,
            CONV_3_0_IN_BIT,
            CONV_3_0_OFM_CH,
            CONV_3_0_OUT_BIT,
            CONV_3_0_W_BIT,
            32,
            CONV_3_0_BIAS_BIT,
            CONV_3_0_SIMD,
            CONV_3_0_PE,
            CONV_3_0_R_SHIFT>(
        conv_2_0_out,
        conv_3_0_w,
        conv_3_0_bias,
        conv_3_0_out,
        reps);

    // conv_4_0
    hls::stream<ap_uint<CONV_4_0_OUT_BIT * CONV_4_0_OFM_CH> > conv_4_0_out("conv_4_0_out");
#pragma HLS STREAM variable=conv_4_0_out depth=32 dim=1

    conv3x3_bn_act<
            CONV_4_0_K,
            CONV_4_0_S,
            CONV_4_0_P,
            CONV_4_0_IFM_ROW,
            CONV_4_0_IFM_COL,
            CONV_4_0_IFM_CH,
            CONV_4_0_IN_BIT,
            CONV_4_0_OFM_CH,
            CONV_4_0_OUT_BIT,
            CONV_4_0_W_BIT,
            32,
            CONV_4_0_BIAS_BIT,
            CONV_4_0_SIMD,
            CONV_4_0_PE,
            CONV_4_0_R_SHIFT>(
        conv_3_0_out,
        conv_4_0_w,
        conv_4_0_bias,
        conv_4_0_out,
        reps);

    // output
    hls::stream<ap_uint<64> >  net_out("net_out");
#pragma HLS STREAM variable=net_out depth=32 dim=1
    StreamingDataWidthConverter_Batch<CONV_4_0_OUT_BIT * CONV_4_0_OFM_CH, 64, CONV_4_0_OFM_ROW*CONV_4_0_OFM_COL> (conv_4_0_out, net_out, reps);

    AddLast<CONV_4_0_OFM_ROW*CONV_4_0_OFM_COL>(net_out, out, reps);
}
void ultra_net(stream<my_ap_axis> & in, stream<my_ap_axis> & out, const unsigned int reps) {

#pragma HLS INTERFACE axis register both port=out
#pragma HLS INTERFACE axis register both port=in
#pragma HLS INTERFACE s_axilite port=reps bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#pragma HLS ARRAY_PARTITION variable = conv_0_0_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_0_0_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_0_1_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_0_1_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_1_0_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_1_0_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_2_0_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_2_0_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_3_0_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_3_0_bias complete dim = 1

#pragma HLS ARRAY_PARTITION variable = conv_4_0_w complete dim = 1
#pragma HLS ARRAY_PARTITION variable = conv_4_0_bias complete dim = 1

    do_compute(in, out, reps);
}
