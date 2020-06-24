//Give the input and output file names on the command line
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sndfile.h>
#include <iostream>
#include <cstring>
#include <vector>
#include <complex.h>
#include <fftw3.h>

using namespace std;

#define ARRAY_LEN(x)    ((int) (sizeof (x) / sizeof (x [0])))
#define MAX(x,y)        ((x) > (y) ? (x) : (y))
#define MIN(x,y)        ((x) < (y) ? (x) : (y))
typedef struct {
    vector<double> next_output_buffer;
    vector<double> overlap_buffer;
    int original_block_size;
    int padded_block_size;
} rt_output_buffers;

double transform(double);
fftw_complex* fftw_complex_transform(vector<double >);
//fftw_complex* ifft_from_complex_vector(vector<vector<double> >);
double* ifft_from_complex_vector(vector<vector<double> >);
vector<double> convolved_block(vector<double>, vector<complex<double> >, int);
vector<complex<double> > filter_kernel_cpx_vec(vector<double>);
vector<double> padded(vector<double>, int);
void testDummyData();
rt_output_buffers yield_processed_buffers(vector<double>, vector<complex<double> >, vector<double>, int, int, int);

int main (int argc, char ** argv) {
    SNDFILE *infile, *outfile, *irfile ;
    SF_INFO sfinfo ;
    double buffer[1024];
    sf_count_t count;

    //for fftw3
    vector<double> inputWavArr;
    vector<double> irWavArr;
    
    if (argc != 4) {
        printf("\nUsage :\n\n    <executable name>  <input signal file> <impulse response file> <output file>\n") ;
        exit(0);
    }
    
    memset (&sfinfo, 0, sizeof (sfinfo)) ;
    if ((infile = sf_open (argv [1], SFM_READ, &sfinfo)) == NULL)     {     
        printf ("Error : Not able to open input file '%s'\n", argv [1]);
        sf_close (infile);
        exit (1) ;
    } 

    if ((irfile = sf_open (argv [2], SFM_READ, &sfinfo)) == NULL)     {     
        printf ("Error : Not able to open input file '%s'\n", argv [2]);
        sf_close (irfile);
        exit (1) ;
    }   
    
    if ((outfile = sf_open (argv [3], SFM_WRITE, &sfinfo)) == NULL) { 
        printf ("Error : Not able to open output file '%s'\n", argv [3]);
        sf_close (outfile);
        exit (1);
    }
    
    while ((count = sf_read_double (infile, buffer, ARRAY_LEN (buffer))) > 0) {
        for (int i = 0; i < 1024; i++)
            inputWavArr.push_back(buffer[i]);
    }
    cout << "input length " << inputWavArr.size() << "\n";
    double sumIrImpulses = 0;
    while ((count = sf_read_double (irfile, buffer, ARRAY_LEN (buffer))) > 0) {
        for (int i = 0; i < 1024; i++) {
            double el = buffer[i];
            irWavArr.push_back(el); // max value 0.0408325
            sumIrImpulses += (el);
        }
    }
    cout << "sumIrImpulses " << sumIrImpulses << "\n";
    cout << "ir length " << irWavArr.size() << "\n";
    sumIrImpulses = abs(sumIrImpulses);
    //const int irLen = 189440; // filter(ir) block len
    int irLen = irWavArr.size();
    //const int windowSize = 72705; // s.t. irLen+windowSize-1 is a pwr of 2
    int windowSize = 262144 - irLen+1; //  262144 is a pwr of 2
    cout << "windowSize " << windowSize << "\n";
    const int outputLength = irLen + windowSize - 1;
    cout << "outputLength " << outputLength << "\n";
    
    sf_close(infile);
    sf_close(irfile);

    irWavArr = padded(irWavArr, outputLength);
    int newIrLength = irWavArr.size();
    
    vector<complex<double> > ir_vec;
    ir_vec = filter_kernel_cpx_vec(irWavArr);

    int numSections = floor(inputWavArr.size()/windowSize);
    if (numSections*windowSize != inputWavArr.size()) {
        inputWavArr = padded(inputWavArr, (numSections+1)*windowSize);
        numSections++;
    }
    
    

    // OVERLAP-ADD PROCEDURE
    vector<vector<double> > totals;
    cout << "numSections is " << numSections << "\n";
    int overlap_length = outputLength - windowSize;
    vector<double> overlap_ring_buffer(overlap_length,0); 
    vector<double> results;
    for (int j=0; j<numSections; j++) { // may be OBOB use numSections+1? or pad inputWavArr? 
        vector<double> total;
        cout << "convolving section " << j << "\n";
        vector<double> section_arr;
        for (int i=j*windowSize; i<(j*windowSize + windowSize); i++) {
            section_arr.push_back(inputWavArr[i]);
        }
        
        
        rt_output_buffers process_results = yield_processed_buffers(section_arr, ir_vec, overlap_ring_buffer, overlap_length, windowSize, outputLength);
        // we don't have TOTALS... just overlap buffer
        //for (int i=0; i<output.size(); i++) {
        //    total.push_back(output[i]); // normalize
       // }
        //totals.push_back(total);
        vector<double> out_buff = process_results.next_output_buffer;
        for (int k=0; k<out_buff.size(); k++) {
            results.push_back(out_buff[k]/150);
        }
        overlap_ring_buffer.clear();
        overlap_ring_buffer = process_results.overlap_buffer;


    }
    //vector<double> results(inputWavArr.size()+newIrLength-1, 0);
    /*
    for (int j=0; j<numSections; j++) {
        vector<double> totals_arr = totals[j];
        cout << "overlap summing section " << j << "\n";
        for (int i=0; i<totals_arr.size(); i++) {
            int newIdx = j*windowSize+i;
            results[newIdx] += totals_arr[i]/550;
        }
    }
    double maxVal = 0;
    for (int i=0; i<results.size(); i++) {
        if (results[i] > maxVal) {
            maxVal = results[i];
        }
    }
    cout << "maxval" << maxVal << "\n";
    cout << "sumIrImpulses" << sumIrImpulses << "\n";
    // RESULTS MARK THE END OF OVERLAP-ADD PROCEDURE
    */
    double* buff3 = (double*)malloc(results.size()*sizeof(double));
    double max = 0;
    for (int idx = 0; idx < results.size(); idx++) {
        double u = transform(results[idx]/150); // tried 550, but max val was .1
        if (u>max) {
            max = u;
        }
        
        buff3[idx] = u; // NO LONGER normalizing factor for these samples... output without this has amplitude going almost up to 120. input file had max around 0.15. max should be 1 about
    }

    cout << "max " << max << "\n";
    free(buff3);
    long writtenFrames = sf_writef_double (outfile, buff3, results.size());
    sf_close (outfile);
    
    return 0 ;
}

rt_output_buffers yield_processed_buffers(vector<double> rt_input_block, vector<complex<double> > filter_kernel, vector<double> overlap_buffer, int overlap_length, int block_size, int block_size_w_padding) {
    rt_output_buffers result_struct;
    result_struct.original_block_size = block_size;
    result_struct.padded_block_size = block_size_w_padding;

        

        // hanning window
        for (int i = 0; i < block_size; i++) {
            double multiplier = 0.5 * (1 - cos(2*M_PI*i/(block_size-1)));
            rt_input_block[i] = multiplier * rt_input_block[i];
        }
        
        rt_input_block = padded(rt_input_block, block_size_w_padding);
        vector<double> output = convolved_block(rt_input_block, filter_kernel, block_size_w_padding);
        // this is not 
        for (int i=0; i<block_size_w_padding-block_size; i++) {
            // BAD PERFORMANCE BELOW... USE RING BUFFER
            output[i] += overlap_buffer[i];
        }
        vector<double> new_overlap_buffer;
        for (int i=block_size; i<block_size_w_padding; i++) {
            new_overlap_buffer.push_back(output[i]);
        }
    vector<double> output_no_overlap;
    for (int k=0; k<block_size; k++) {
        output_no_overlap.push_back(output[k]);
    }
    //result_struct.next_output_buffer = output;
    result_struct.next_output_buffer = output_no_overlap;
    result_struct.overlap_buffer = new_overlap_buffer;
    return result_struct;
}

fftw_complex* fftw_complex_transform(vector<double> signal_wav) {
    int N = signal_wav.size();
    double *in;
    fftw_complex *out;
    fftw_plan irPlan;
    //in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N);
    in = (double*) malloc(sizeof(double)*N);
    out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N);
    
    for (int i = 0; i < signal_wav.size(); i++)
    {
        //in[i][0] = signal_wav[i];
        in[i] = signal_wav[i];
        //in[i][1] = (float)0; // complex component .. 0 for input of wav file
    }
    //irPlan = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    irPlan = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
    fftw_execute(irPlan);
    fftw_destroy_plan(irPlan);
    fftw_free(in);
    
    return out;
}

vector<complex<double> > filter_kernel_cpx_vec(vector<double> input) {
    fftw_complex* irFFT = fftw_complex_transform(input);

    vector<complex<double> > kernel_vec;
    for (int i=0; i<input.size(); i++) {
        complex<double> m1 (irFFT[i][0], irFFT[i][1]);
        kernel_vec.push_back(m1);
    }

    fftw_free(irFFT); 
    return kernel_vec;
}

//fftw_complex* ifft_from_complex_vector(vector<vector<double> > signal_vec) {
double* ifft_from_complex_vector(vector<vector<double> > signal_vec) {
    int N = signal_vec.size();
    //fftw_complex *in, *out;
    fftw_complex *in;
    double *out;
    fftw_plan irPlan;
    in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N);
    //out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*N);
    out = (double*) malloc(sizeof(double)*N);
    
    for (int i = 0; i < signal_vec.size(); i++)
    {
        in[i][0] = signal_vec[i][0]; // real
        in[i][1] = signal_vec[i][1]; // imag
    }
    //irPlan = fftw_plan_dft_1d(N, in, out, FFTW_BACKWARD, FFTW_ESTIMATE);
    irPlan = fftw_plan_dft_c2r_1d(N, in, out, FFTW_ESTIMATE);
    fftw_execute(irPlan);
    fftw_destroy_plan(irPlan);
    fftw_free(in);
    
    return out;
}

vector<double> convolved_block(vector<double> in_vector, vector<complex<double> > ir_cpx_vector, int size) {
    fftw_complex* outputFFT = fftw_complex_transform(in_vector);

    vector<vector<double> > products;
    vector<complex<double> > sig_fft_cpx;
    for (int i=0; i<size; i++) {
        complex<double> m1 (outputFFT[i][0], outputFFT[i][1]);
        sig_fft_cpx.push_back(m1);
    }        
    fftw_free(outputFFT);
    for (int j=0; j<size; j++) {
        std::complex<double> complexProduct = sig_fft_cpx[j]*ir_cpx_vector[j];
        double re = real(complexProduct);
        double im = imag(complexProduct);
        vector<double> elemVec(2);
        elemVec[0] = re;
        elemVec[1] = im;
        
        products.push_back(elemVec);
    }
    //fftw_complex* revFFT = ifft_from_complex_vector(products);
    double* revFFT = ifft_from_complex_vector(products);
    vector<double> out_vec_dbl;
    
    for (int i=0; i<size; i++) {
        //out_vec_dbl.push_back(outputFFT[i][0]);
        //out_vec_dbl.push_back(revFFT[i][0]);
        out_vec_dbl.push_back(revFFT[i]);
        //out_vec_dbl.push_back(outputFFT[i]);
    }
    //fftw_free(revFFT);
    free(revFFT);
    return out_vec_dbl;
}

vector<double> padded(vector<double> input, int total) {
    vector<double> padded_vec;
    for (int i = 0; i<input.size(); i++) {
    padded_vec.push_back(input[i]);
    }
    int numZeroes = total - input.size();
    for (int i = 0; i< numZeroes; i++) {
    padded_vec.push_back((double)0);
    }
    return padded_vec;
}

void testDummyData() {
    vector<double> dummyFilter;
    dummyFilter.push_back(1);
    dummyFilter.push_back(-1);
    dummyFilter.push_back(1);

    vector<double> dummySignal;
    dummySignal.push_back(3);
    dummySignal.push_back(-1);
    dummySignal.push_back(0);
    dummySignal.push_back(3);
    dummySignal.push_back(2);
    dummySignal.push_back(0);
    dummySignal.push_back(1);
    dummySignal.push_back(2);
    dummySignal.push_back(1);

    const int nearWindowSize=3;
    const int nearIrLength=3;
    const int totalLength = 5;

    dummyFilter = padded(dummyFilter, totalLength);
    vector<complex<double> > dummy_ir_vec = filter_kernel_cpx_vec(dummyFilter);

    int localNumSections = 3;
    vector<vector<double> > outputs;
    for (int j=0; j<localNumSections; j++) {
        vector<double> local_section;
        for (int i; i<nearWindowSize; i++) {
            int idx = j*nearWindowSize + i;
            local_section.push_back(dummySignal[idx]);
        }
        local_section = padded(local_section, totalLength);
        vector<double> local_output = convolved_block(local_section, dummy_ir_vec, totalLength);
        outputs.push_back(local_output);
    }
    vector<double> local_results(11,0); // example has 11 in output
    
    for (int j=0; j<localNumSections; j++) {
        vector<double> local_totals_arr = outputs[j];
        cout << "overlap summing section " << j << "\n";
        for (int i=0; i<local_totals_arr.size(); i++) {
            int newIdx = j*nearWindowSize+i;
            local_results[newIdx] += local_totals_arr[i];
        }
    }    
    for (int i=0; i<11; i++) {
        cout << "result " << i << "\n";
        cout << local_results[i] << "\n";
    }
}

double transform(double n) {
   // return n + 12*pow(n,4);
    double u = n/4;
    return u;
    //return myPow*2;
}