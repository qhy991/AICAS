#pragma once

#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;

#include "sliding_window_unit.h"
#include "stream_tools.h"


/**
 * maxpool层计算处理函数
 */
template <	unsigned K,
			unsigned IN_CH,
			unsigned IN_BIT,
            unsigned VEC_NUMS>
void maxpool_cal(
	stream<ap_uint<IN_CH*IN_BIT> >& vec,
	stream<ap_uint<IN_CH*IN_BIT> >& out,
	const unsigned reps = 1)
{
	ap_uint<IN_CH*IN_BIT> result = 0;
	unsigned k_cnt = 0;

	for (unsigned rep = 0; rep < reps*VEC_NUMS; rep++) {
#pragma HLS PIPELINE II=1

        // 这里的temp_vec应该是寄存器（reg）类型
		ap_uint<IN_CH*IN_BIT> temp_vec = vec.read();

		for (unsigned c = 0; c < IN_CH; c++) {
#pragma HLS UNROLL
            // if(temp_vec((c+1)*IN_BIT-1, c*IN_BIT) > result( (c+1)*IN_BIT-1, c*IN_BIT)) {
            //     result( (c+1)*IN_BIT-1, c*IN_BIT) = temp_vec((c+1)*IN_BIT-1, c*IN_BIT);
            // }

			ap_uint<IN_BIT> temp = temp_vec( (c+1)*IN_BIT-1 , c*IN_BIT );
				
			result( (c+1)*IN_BIT-1, c*IN_BIT ) = (temp > result( (c+1)*IN_BIT-1, c*IN_BIT )) ? temp : result( (c+1)*IN_BIT-1, c*IN_BIT );
		}

        if(++ k_cnt == K*K) {
            out.write(result);
            result = 0;
            k_cnt = 0;
        }
	}
}

/**
 * avgpool层计算处理函数
 * 默认k=2
 */
template <	unsigned K,
			unsigned IN_CH,
			unsigned IN_BIT,
            unsigned VEC_NUMS>
void avgpool_cal(
	stream<ap_uint<IN_CH*IN_BIT> >& vec,
	stream<ap_uint<IN_CH*IN_BIT> >& out,
	const unsigned reps = 1)
{
	ap_uint<IN_CH*IN_BIT+1> temp_result = 0;  // Increase 1 bit to prevent overflow
	unsigned k_cnt = 0;

	for (unsigned rep = 0; rep < reps*VEC_NUMS; rep++) {
#pragma HLS PIPELINE II=1

        // 这里的temp_vec应该是寄存器（reg）类型
		ap_uint<IN_CH*IN_BIT> temp_vec = vec.read();

		for (unsigned c = 0; c < IN_CH; c++) {
#pragma HLS UNROLL

			ap_uint<IN_BIT> temp = temp_vec( (c+1)*IN_BIT-1 , c*IN_BIT );

			temp_result( (c+1)*IN_BIT-1, c*IN_BIT ) = temp + temp_result( (c+1)*IN_BIT-1, c*IN_BIT );
				
		}

        if(++ k_cnt == K*K) {
			ap_uint<IN_CH*IN_BIT> result = temp_result >> 2;  // result = result / (K*K)
            out.write(result);
            temp_result = 0;
            k_cnt = 0;
        }
	}
}


/*
* 最大池化层
* 支持kernel=2,stride=2
*/
template <	unsigned K,                 // kernel
			// unsigned S,                 // stride
			unsigned IN_ROW,
			unsigned IN_COL,
			unsigned IN_CH,
			unsigned IN_BIT>
void max_pool2d(
	stream<ap_uint<IN_CH*IN_BIT> >& in,
	stream<ap_uint<IN_CH*IN_BIT> >& out, 
	const unsigned reps = 1)
{
#pragma HLS DATAFLOW
    // TODO IN_ROW % S != 0
    // 暂时只处理特殊情况
    const unsigned OUT_ROW = IN_ROW / 2;
    const unsigned OUT_COL = IN_COL / 2; 
    const unsigned S = 2;

    // 产生滑动窗口数据
    hls::stream<ap_uint<IN_CH*IN_BIT> > swu_out("swu_out");
    SWU<K, S, IN_ROW, IN_COL, IN_CH, IN_BIT>(in, swu_out, reps);

    // 处理数据
    maxpool_cal<K, IN_CH, IN_BIT, OUT_ROW*OUT_COL*K*K>(swu_out, out, reps);
}


/*
* 平均池化层
* 支持kernel=2,stride=2
*/
template <	unsigned K,                 // kernel
			// unsigned S,                 // stride
			unsigned IN_ROW,
			unsigned IN_COL,
			unsigned IN_CH,
			unsigned IN_BIT>
void avg_pool2d(
	stream<ap_uint<IN_CH*IN_BIT> >& in,
	stream<ap_uint<IN_CH*IN_BIT> >& out, 
	const unsigned reps = 1)
{
#pragma HLS DATAFLOW
    // TODO IN_ROW % S != 0
    // 暂时只处理特殊情况
    const unsigned OUT_ROW = IN_ROW / 2;
    const unsigned OUT_COL = IN_COL / 2; 
    const unsigned S = 2;

    // 产生滑动窗口数据
    hls::stream<ap_uint<IN_CH*IN_BIT> > swu_out("swu_out");
    SWU<K, S, IN_ROW, IN_COL, IN_CH, IN_BIT>(in, swu_out, reps);

    // 处理数据
    avgpool_cal<K, IN_CH, IN_BIT, OUT_ROW*OUT_COL*K*K>(swu_out, out, reps);
}