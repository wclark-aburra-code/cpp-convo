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

//using namespace std;
// FOR MAKE_UNIQUE....
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

// included to use make_unique as sole C++14 feature here
namespace std {
    template<class T> struct _Unique_if {
        typedef unique_ptr<T> _Single_object;
    };

    template<class T> struct _Unique_if<T[]> {
        typedef unique_ptr<T[]> _Unknown_bound;
    };

    template<class T, size_t N> struct _Unique_if<T[N]> {
        typedef void _Known_bound;
    };

    template<class T, class... Args>
        typename _Unique_if<T>::_Single_object
        make_unique(Args&&... args) {
            return unique_ptr<T>(new T(std::forward<Args>(args)...));
        }

    template<class T>
        typename _Unique_if<T>::_Unknown_bound
        make_unique(size_t n) {
            typedef typename remove_extent<T>::type U;
            return unique_ptr<T>(new U[n]());
        }

    template<class T, class... Args>
        typename _Unique_if<T>::_Known_bound
        make_unique(Args&&...) = delete;
}

// clean up...
#define ARRAY_LEN(x)    ((int) (sizeof (x) / sizeof (x [0])))
#define MAX(x,y)        ((x) > (y) ? (x) : (y))
#define MIN(x,y)        ((x) < (y) ? (x) : (y))

double hanning_multiplier(const double i, const double block_size) {
    return 0.5 * (1 - cos(2*M_PI*i/(block_size-1)));
}

template <class T>
class DspBuffer {

    public:    
	DspBuffer(const int size_) : 
	vec(size_, 0), used(-1), nuevo(-1), end(-1), size(size_)
    {
    }
    void insert(T el) {
		end = increment(end);
		vec[end] = el;
	}
	void reset() {
        used = -1;
		nuevo = end;
	}
    T next() {
		if (nuevo > used) {
			used = increment(used);
            return vec[used]; // “some”
		}
		else {
            return 0; // store "zero value"... for complex numbers
		}
	}
	std::vector<T> get_vec() { // not called?
		return vec;
	}
	void clear() {
		used = -1;
		nuevo = -1;
		end = -1;
	}
private:
    int increment(const int index) { // helper function
      if (index + 1 == size) {
        throw std::invalid_argument("out of bounds index");
      }
        return (index + 1);
    }
    mutable std::vector<T> vec;
	mutable int used;
	mutable int nuevo;
	mutable int end;
	const int size;
};

struct FFTWDeleterFunctor {  
    void operator()(fftw_complex* fft) {
        fftw_free(fft);
    }
};

struct SendfileDeleterFunctor {  
    void operator()(SNDFILE * snd_file) {
        sf_close(snd_file);
    }
};

std::unique_ptr<fftw_complex, FFTWDeleterFunctor> fftw_complex_transform(const std::vector<double>&);
std::vector<std::complex<double> > filter_kernel_cpx_vec(const std::vector<double>&);
std::vector<double> padded(const std::vector<double>&, const int);

int main (int argc, char ** argv) {
    SF_INFO sfinfo ;
    double file_buffer[1024];
    sf_count_t count;

    //for fftw3
    std::vector<double> input_wav_arr;
    std::vector<double> ir_wav_arr;
    
    if (argc != 4) {
        printf("\nUsage :\n\n    <executable name>  <input signal file> <impulse response file> <output file>\n") ;
        exit(0);
    }
    std::unique_ptr<SNDFILE, SendfileDeleterFunctor> infile(sf_open (argv [1], SFM_READ, &sfinfo));
    std::unique_ptr<SNDFILE, SendfileDeleterFunctor> irfile(sf_open (argv [2], SFM_READ, &sfinfo));
    std::unique_ptr<SNDFILE, SendfileDeleterFunctor> outfile(sf_open (argv [3], SFM_WRITE, &sfinfo));
    memset (&sfinfo, 0, sizeof (sfinfo));   
    if (infile == nullptr)     {        
        printf ("Error : Not able to open input file '%s'\n", argv [1]);
       return 1;
    } 

    if (irfile == nullptr)     {        
        printf ("Error : Not able to open input file '%s'\n", argv [2]);
       return 1;
    }
    if (outfile == nullptr)     {        
        printf ("Error : Not able to open output file '%s'\n", argv [3]);
        return 1;
    }
    
    while ((count = sf_read_double (infile.get(), file_buffer, ARRAY_LEN (file_buffer))) > 0) {
        for (int i = 0; i < 1024; i++)
            input_wav_arr.push_back(file_buffer[i]);
    }
    std::cout << "input length " << input_wav_arr.size() << "\n";
    while ((count = sf_read_double (irfile.get(), file_buffer, ARRAY_LEN (file_buffer))) > 0) {
        for (int i = 0; i < 1024; i++) {
            double el = file_buffer[i];
            ir_wav_arr.push_back(el); // max value 0.0408325
        }
    }
    std::cout << "ir length " << ir_wav_arr.size() << "\n";
    
    // end file-reading procedure

    int ir_length = ir_wav_arr.size();
    //const int windowSize = 72705; // s.t. irLen+windowSize-1 is a pwr of 2
    int total_size = 262144;
    while (total_size < ir_length) { // in case the IR is extra long; btw we aren't handling small IR's well...
        total_size *= 2;
    }
    int window_size = total_size - ir_length + 1; //  262144 is a pwr of 2
    std::cout << "window_size " << window_size << "\n"; // need a compile-time const #define injected here
    const int output_length = total_size;
    std::cout << "output_length " << output_length << "\n";

    ir_wav_arr = padded(ir_wav_arr, output_length);
    
    std::vector<std::complex<double> > ir_vec = filter_kernel_cpx_vec(ir_wav_arr);

    int num_sections = floor(input_wav_arr.size()/window_size);
    if (num_sections*window_size != input_wav_arr.size()) { // likely they don't happen to be equal
        input_wav_arr = padded(input_wav_arr, (num_sections+1)*window_size);
        num_sections++;
    }

    // OVERLAP-ADD PROCEDURE
    std::cout << "num_sections is " << num_sections << "\n";
    int overlap_length = output_length - window_size;
    double window_size_float = double(window_size);
    std::complex<double> m1;

    std::vector<double> results;
    results.reserve(input_wav_arr.size() + ir_wav_arr.size() - 1);

    DspBuffer<double> section_buffer(output_length);
    DspBuffer<double> overlap_buffer(overlap_length);
    DspBuffer<double> new_overlap_buffer(overlap_length);
    DspBuffer<std::complex<double> > sig_fft_complex(output_length);
    DspBuffer<double> output_buffer(window_size);
    DspBuffer<double> product_reals(output_length);
    DspBuffer<double> product_imaginaries(output_length);
    std::complex<double> complex_product;

    double re;
    double im;

    auto in_cpx = std::make_unique<double[]>(output_length);
    std::unique_ptr<fftw_complex, FFTWDeleterFunctor> out_cpx((fftw_complex*) fftw_malloc(sizeof(fftw_complex)*output_length));
    fftw_plan real_to_complex = fftw_plan_dft_r2c_1d(output_length, in_cpx.get(), out_cpx.get(), FFTW_ESTIMATE);

    std::unique_ptr<fftw_complex, FFTWDeleterFunctor> in((fftw_complex*) fftw_malloc(sizeof(fftw_complex)*output_length));
    auto out = std::make_unique<double[]>(output_length);
    fftw_plan complex_to_real = fftw_plan_dft_c2r_1d(output_length, in.get(), out.get(), FFTW_ESTIMATE);

    double i_float;
    int i;
    for (int j=0; j<num_sections; j++) { // may be OBOB use num_sections+1? or pad inputWavArr?       
        // comment out the following for super speed...
        std::cout << "convolving section " << j << "\n";
        
        i_float = 0; // for hanning multiplier, we increment this in parallel to int
        for ( i=j*window_size; i<(j*window_size + window_size); i++) {
            //section_arr.push_back(inputWavArr[i]);
            section_buffer.insert(hanning_multiplier(i_float, window_size_float) * input_wav_arr[i]);
            i_float++;
        }
        // padding - we avoid call to padded() because that contains call to "new"
        for ( i=window_size; i<output_length; i++) {
            section_buffer.insert(0.0);
        }
        section_buffer.reset();
        for (i = 0; i < output_length; i++)
        {
            in_cpx[i] = section_buffer.next();
        }

        fftw_execute(real_to_complex);

        for (i=0; i<output_length; i++) {
            m1.imag(out_cpx.get()[i][1]);
            m1.real(out_cpx.get()[i][0]);
            sig_fft_complex.insert(m1);
        }     
        sig_fft_complex.reset();   
        for (i=0; i<output_length; i++) {
            complex_product = sig_fft_complex.next()*ir_vec[i];
            re = real(complex_product);
            im = imag(complex_product);
            product_reals.insert(re);
            product_imaginaries.insert(im);
        }
        product_reals.reset();
        product_imaginaries.reset();

        for (i = 0; i < output_length; i++)
        {
            in.get()[i][0] = product_reals.next(); // real
            in.get()[i][1] = product_imaginaries.next(); // imag
        }
        fftw_execute(complex_to_real);
        overlap_buffer.reset();
        new_overlap_buffer.clear(); // ONLY USE OF CLEAR ASIDE FROM END OF THIS BIG ITER BLOCK
        for (i=0; i<window_size; i++) {
            output_buffer.insert(out.get()[i]+overlap_buffer.next());
        }
        for (i = window_size; i < output_length; i++) {
            if (i < overlap_length) {
                new_overlap_buffer.insert(out.get()[i]+overlap_buffer.next());
            }
            else {
                new_overlap_buffer.insert(out.get()[i]);
            }
        }
        overlap_buffer.clear();
        new_overlap_buffer.reset();
        for (i=0; i<overlap_length; i++) {
            overlap_buffer.insert(new_overlap_buffer.next());
        } 
        overlap_buffer.reset();
        output_buffer.reset();
        for (i=0; i < window_size; i++) {
            results.push_back(output_buffer.next());
        }
        // CLEAR ALL BUFFERS EXCEPT OVERLAP BUFFER
        section_buffer.clear();
        sig_fft_complex.clear();
        product_reals.clear();
        product_imaginaries.clear(); 
        output_buffer.clear(); 
        // new_overlap buffer and overlap_buffer get cleared at more explicit points prior to here
    }
    auto write_file_buffer = std::make_unique<double[]>(results.size());
    
    for (int idx = 0; idx < results.size(); idx++) {
        write_file_buffer.get()[idx] = results[idx]/200000; // NO LONGER normalizing factor for these samples... output without this has amplitude going almost up to 120. input file had max around 0.15. max should be 1 about
        // why the way bigger normalization factor???
        // HAVE to do more ... was 12050
    }
    fftw_destroy_plan(complex_to_real);
    fftw_destroy_plan(real_to_complex);
    long written_frames = sf_writef_double (outfile.get(), write_file_buffer.get(), results.size());
    return 0 ;
}

std::unique_ptr<fftw_complex, FFTWDeleterFunctor> fftw_complex_transform(const std::vector<double>& signal_wav) {
    int signal_size = signal_wav.size();
    fftw_plan ir_plan; // opaque pointer, explain in comments why we don't wrap in RAII
    auto in = std::make_unique<double[]>(signal_size);
    std::unique_ptr<fftw_complex, FFTWDeleterFunctor> out((fftw_complex*) fftw_malloc(sizeof(fftw_complex)*signal_size));
    for (int i = 0; i < signal_wav.size(); i++)
    {
        in[i] = signal_wav[i]; // don't need to use get?
    }
    ir_plan = fftw_plan_dft_r2c_1d(signal_size, in.get(), out.get(), FFTW_ESTIMATE);
    fftw_execute(ir_plan);
    fftw_destroy_plan(ir_plan);
    return std::move(out);
}

std::vector<std::complex<double> > filter_kernel_cpx_vec(const std::vector<double>& input) {
    std::unique_ptr<fftw_complex, FFTWDeleterFunctor> ir_fft = fftw_complex_transform(input);
    std::vector<std::complex<double> > kernel_vec;
    kernel_vec.reserve(input.size());
    for (int i=0; i<input.size(); i++) {
        std::complex<double> m1 (ir_fft.get()[i][0], ir_fft.get()[i][1]);
        kernel_vec.push_back(m1);
    }
    return kernel_vec;
}

std::vector<double> padded(const std::vector<double>& input, const int total) {
    std::vector<double> padded_vec;
    padded_vec.reserve(total);
    for (int i = 0; i<input.size(); i++) {
        padded_vec.push_back(input[i]);
    }
    int num_zeroes = total - input.size();
    for (int i = 0; i < num_zeroes; i++) {
        padded_vec.push_back(0.0);
    }
    return padded_vec;
}