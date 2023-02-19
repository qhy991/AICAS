#pragma once

#include <ap_int.h>
#include <hls_stream.h>
using namespace hls;
using namespace std;

// axi data
struct my_ap_axis
{
    ap_uint<64> data;
    ap_uint<1> last;
    ap_uint<8> keep;
};

struct my_ap_axis_out
{
    ap_uint<16> data;
    ap_uint<1> last;
    ap_uint<8> keep;
};

template <unsigned NumLines>
void AddLast(stream<ap_uint<64> > &in, stream<my_ap_axis> &out,
             const unsigned reps = 1) {
    my_ap_axis temp;
    temp.keep = 0xff;

    for (unsigned i = 0; i < reps * NumLines - 1; i++) {
        temp.data = in.read();
        temp.last = 0;
        out.write(temp);
    }

    temp.data = in.read();
    temp.last = 1;
    out.write(temp);
}

template <unsigned OutStreamW, unsigned NumLines>
void ExtractPixels(stream<my_ap_axis> &in, stream<ap_uint<OutStreamW> > &out, 
                   const unsigned reps = 1) {
    my_ap_axis temp;

    for (unsigned rep = 0; rep < reps * NumLines; rep++) {
#pragma HLS PIPELINE II = 1
        temp = in.read();
        out.write(temp.data(OutStreamW - 1, 0));
    }
}

template <unsigned int InWidth,  // width of input stream
          unsigned int OutWidth,  // width of output stream
          unsigned int NumInWords>  // number of input words to process
void StreamingDataWidthConverter_Batch(hls::stream<ap_uint<InWidth> > &in,
                                       hls::stream<ap_uint<OutWidth> > &out,
                                       const unsigned int numReps) {
    if (InWidth > OutWidth) {
        // emit multiple output words per input word read
        // CASSERT_DATAFLOW(InWidth % OutWidth == 0);
        const unsigned int outPerIn = InWidth / OutWidth;
        const unsigned int totalIters = NumInWords * outPerIn * numReps;
        unsigned int o = 0;
        ap_uint<InWidth> ei = 0;
        for (unsigned int t = 0; t < totalIters; t++) {
#pragma HLS PIPELINE II = 1
            // read new input word if current out count is zero
            if (o == 0) {
                ei = in.read();
            }
            // pick output word form the rightmost position
            ap_uint<OutWidth> eo = ei(OutWidth - 1, 0);
            out.write(eo);
            // shift input to get new output word for next iteration
            ei = ei >> OutWidth;
            // increment written output count
            o++;
            // wraparound indices to recreate the nested loop structure
            if (o == outPerIn) {
                o = 0;
            }
        }
    }
    else if (InWidth == OutWidth) {
        // stratight-through copy
        for (unsigned int i = 0; i < NumInWords * numReps; i++) {
#pragma HLS PIPELINE II = 1
            ap_uint<InWidth> e = in.read();
            out.write(e);
        }
    }
    else {  // InWidth < OutWidth
        // read multiple input words per output word emitted
        // CASSERT_DATAFLOW(OutWidth % InWidth == 0);
        const unsigned int inPerOut = OutWidth / InWidth;
        const unsigned int totalIters = NumInWords * numReps;
        unsigned int i = 0;
        ap_uint<OutWidth> eo = 0;
        for (unsigned int t = 0; t < totalIters; t++) {
#pragma HLS PIPELINE II = 1
            // read input and shift into output buffer
            ap_uint<InWidth> ei = in.read();
            eo = eo >> InWidth;
            eo(OutWidth - 1, OutWidth - InWidth) = ei;
            // increment read input count
            i++;
            // wraparound logic to recreate nested loop functionality
            if (i == inPerOut) {
                i = 0;
                out.write(eo);
            }
        }
    }
}
