#pragma once
#include <hls_stream.h>
#include <ap_int.h>
using namespace hls;


/*
* 卷积计算单元 同时计算bn层与激活层
* 在矩阵向量计算后立即计算得到激活输出值
* 计算3x3的卷积
* 如k=3, p=1, s=1
* 或k=3, p=2, s=1
* 输入数据宽度为 IN_STREAM_BIT
* 输出数据宽度为 PE * OUT_BIT
*/
template <
            unsigned CONV_K,  // kernel size
            unsigned CONV_S,  // stride
            unsigned CONV_P,  // padding

            unsigned IN_ROW,  // input h
            unsigned IN_COL,  // input w
            unsigned IN_CH,   // input channel
            unsigned IN_BIT,  // input bit
            
            unsigned OUT_CH,  // output channel
            unsigned OUT_BIT,  // output bit
            
            unsigned W_BIT,  // weight bit
            unsigned M_BIT,  // middle bit
            unsigned BIAS_BIT,  // biase bit
            
            unsigned SIMD,
            unsigned PE,
            unsigned R_SHIFT>
void conv3x3_bn_act(
    stream<ap_uint<IN_BIT * IN_CH> > & in,
    const ap_uint<SIMD * W_BIT> weights[PE][((IN_CH*9)/SIMD)*(OUT_CH/PE)],
    const ap_int<BIAS_BIT> bias[PE][OUT_CH/PE],
    stream<ap_uint<OUT_BIT * OUT_CH> > & out,
    const unsigned reps = 1) {
#pragma HLS DATAFLOW
    // padding后输入维�?
    const unsigned INTER_ROW = IN_ROW + 2 * CONV_P;
    const unsigned INTER_COL = IN_COL + 2 * CONV_P;
    // 输出维度
    const unsigned OUT_ROW = (IN_ROW + 2 * CONV_P - CONV_K)/CONV_S + 1;
    const unsigned OUT_COL = (IN_COL + 2 * CONV_P - CONV_K)/CONV_S + 1;

    // padding
    stream<ap_uint<IN_CH*IN_BIT> > padding_out("samepad_out");
#pragma HLS STREAM variable=padding_out depth=2 dim=1
    padding<IN_ROW, IN_COL, IN_CH, IN_BIT, CONV_P> (in, padding_out, reps);
    // sliding windows
    stream<ap_uint<IN_CH*IN_BIT> > swu_out("swu_out");
#pragma HLS STREAM variable=swu_out depth=2 dim=1
    SWU<CONV_K, CONV_S, INTER_ROW, INTER_COL, IN_CH, IN_BIT> (padding_out, swu_out, reps);
    // 位宽调整
    stream<ap_uint<SIMD*IN_BIT> > adj_out("adj_out");
#pragma HLS STREAM variable=adj_out depth=2 dim=1
    StreamingDataWidthConverter_Batch<IN_CH*IN_BIT, SIMD*IN_BIT, 9*OUT_ROW*OUT_COL>(swu_out, adj_out, reps);

    // 矩阵向量计算
    stream<ap_uint<PE*OUT_BIT> > mvau_out("mvau_out");
#pragma HLS STREAM variable=mvau_out depth=2 dim=1
    matrix_vector_act_unit<IN_CH*3*3, OUT_CH, IN_BIT, OUT_BIT, W_BIT, M_BIT, BIAS_BIT, SIMD, PE, R_SHIFT, OUT_ROW*OUT_COL>
    (adj_out, weights, bias, mvau_out, reps);

    StreamingDataWidthConverter_Batch<PE*OUT_BIT, OUT_CH*OUT_BIT, OUT_ROW * OUT_COL * OUT_CH / PE>(mvau_out, out, reps);
}

template <
			unsigned IN_ROW,
			unsigned IN_COL,
			unsigned IN_CH,
			unsigned IN_BIT,

			unsigned OUT_CH,
            unsigned OUT_BIT,

			unsigned W_BIT,
			unsigned M_BIT,
            unsigned BIAS_BIT,

			unsigned SIMD,
			unsigned PE,
            unsigned R_SHIFT>
void conv1x1_bn_act(
	stream<ap_uint<IN_BIT * IN_CH> >& in, 
	const ap_uint<SIMD*W_BIT> weights[PE][((IN_CH*1)/SIMD)*(OUT_CH/PE)],
	const ap_int<BIAS_BIT> bias[PE][OUT_CH/PE],
    stream<ap_uint<OUT_BIT * OUT_CH> >& out, 
	const unsigned reps = 1)
{
#pragma HLS DATAFLOW
	const unsigned OUT_ROW = IN_ROW;
	const unsigned OUT_COL = IN_COL;

    // 位宽调整
    stream<ap_uint<SIMD*IN_BIT> > adj_out("adj_out");
    StreamingDataWidthConverter_Batch<IN_CH*IN_BIT, SIMD*IN_BIT, OUT_ROW*OUT_COL>(in, adj_out, reps);

    // 矩阵向量计算
    stream<ap_uint<PE*OUT_BIT> > mvau_out("mvau_out");
	matrix_vector_act_unit<IN_CH, OUT_CH, IN_BIT, OUT_BIT, W_BIT, M_BIT, BIAS_BIT, SIMD, PE, R_SHIFT, OUT_ROW*OUT_COL>
	(adj_out, weights, bias, mvau_out, reps);

    StreamingDataWidthConverter_Batch<PE*OUT_BIT, OUT_CH*OUT_BIT, OUT_ROW * OUT_COL * OUT_CH / PE>(mvau_out, out, reps);
    
}
