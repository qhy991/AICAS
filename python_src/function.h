#ifdef DEBUG_function
#include <iostream>
using namespace std;

#endif

#pragma once
#include <hls_stream.h>
#include <ap_int.h>

using namespace std;
#include <assert.h>
#include "stream_tools.h"


/*
* padding 函数
*/
template <  unsigned IN_ROW,
            unsigned IN_COL,
            unsigned IN_CH,
            unsigned IN_BIT,
            unsigned P>
void padding(
    // 将每一数竖看成一个元素
    stream<ap_uint<IN_CH*IN_BIT> > &in,
    stream<ap_uint<IN_CH*IN_BIT> > &out,
    const unsigned reps = 1) 
{
    const unsigned OUT_ROW = IN_ROW + 2 * P;
    const unsigned OUT_COL = IN_COL + 2 * P;

    ap_uint<IN_CH*IN_BIT> temp_out = 0;

    for (unsigned rep = 0; rep < reps; rep++) {

        for (unsigned h = 0; h < P; h++) {
            for (unsigned s = 0; s < OUT_COL; s++) {
                out.write(0);
            }
        }

        for (unsigned h = 0; h < IN_ROW; h++) {
            for (unsigned s = 0; s < OUT_COL; s++) {
#pragma HLS PIPELINE II = 1
                if ((s < P) || (s >= OUT_COL-P)) {
                    temp_out = 0;
                }
                else {
                    temp_out = in.read();
                }

                out.write(temp_out);
            }
        }

        for (unsigned h = 0; h < P; h++) {
            for (unsigned i = 0; i <OUT_COL; i++) {
                out.write(0);
            }
        }
    }
}


template <	unsigned IN_BIT,
			unsigned OUT_BIT,
			unsigned BIAS_BIT,

			unsigned R_SHIFT>
ap_uint<OUT_BIT> bn_qurelu( ap_int<IN_BIT> in,
                ap_int<BIAS_BIT> bias ) {

    ap_int<IN_BIT> bn_res = in + bias;
    ap_uint<OUT_BIT> res;

   if (bn_res > 0) {
       bn_res = bn_res >> R_SHIFT;
       if (bn_res > 255) {
           res = 255;
       }
       else
           res = bn_res;
   }
   else
       res = 0;
    
    #ifdef DEBUG_function
        cout << "bn_res "  << bn_res << " res " << res << endl;
    #endif

	return res;
    
}
