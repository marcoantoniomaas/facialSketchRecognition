#include "descriptors.hpp"

int  UniformPattern59[256] = {
	1,   2,   3,   4,   5,   0,   6,   7,   8,   0,   0,   0,   9,   0,  10,  11,
	12,   0,   0,   0,   0,   0,   0,   0,  13,   0,   0,   0,  14,   0,  15,  16,
	17,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
	18,   0,   0,   0,   0,   0,   0,   0,  19,   0,   0,   0,  20,   0,  21,  22,
	23,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
	0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
	24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
	25,   0,   0,   0,   0,   0,   0,   0,  26,   0,   0,   0,  27,   0,  28,  29,
	30,  31,   0,  32,   0,   0,   0,  33,   0,   0,   0,   0,   0,   0,   0,  34,
	0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  35,
	0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
	0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  36,
	37,  38,   0,  39,   0,   0,   0,  40,   0,   0,   0,   0,   0,   0,   0,  41,
	0,    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  42,
	43,  44,   0,  45,   0,   0,   0,  46,   0,   0,   0,   0,   0,   0,   0,  47,
	48,  49,   0,  50,   0,   0,   0,  51,  52,  53,   0,  54,  55,  56,  57,  58
};


//------------------------------------------------------------------------------
// elbp
//------------------------------------------------------------------------------
template <typename _Tp> static
inline void elbp_(InputArray _src, OutputArray _dst, int radius, int neighbors){
	//get matrices
	Mat src = _src.getMat();
	// allocate memory for result
	_dst.create(src.rows-2*radius, src.cols-2*radius, CV_32SC1);
	Mat dst = _dst.getMat();
	// zero
	dst.setTo(0);
	for(int n=0; n<neighbors; n++){
		// sample points
		float x = static_cast<float>(-radius) * sin(2.0*CV_PI*n/static_cast<float>(neighbors));
		float y = static_cast<float>(radius) * cos(2.0*CV_PI*n/static_cast<float>(neighbors));
		// relative indices
		int fx = static_cast<int>(floor(x));
		int fy = static_cast<int>(floor(y));
		int cx = static_cast<int>(ceil(x));
		int cy = static_cast<int>(ceil(y));
		// fractional part
		float ty = y - fy;
		float tx = x - fx;
		// set interpolation weights
		float w1 = (1 - tx) * (1 - ty);
		float w2 =      tx  * (1 - ty);
		float w3 = (1 - tx) *      ty;
		float w4 =      tx  *      ty;
		// iterate through your data
		for(int i=radius; i < src.rows-radius;i++){
			for(int j=radius;j < src.cols-radius;j++){
				// calculate interpolated value
				float t = w1*src.at<_Tp>(i+fy,j+fx) + w2*src.at<_Tp>(i+fy,j+cx) + w3*src.at<_Tp>(i+cy,j+fx) + w4*src.at<_Tp>(i+cy,j+cx);
				// floating point precision, so check some machine-dependent epsilon
				dst.at<int>(i-radius,j-radius) += ((t > src.at<_Tp>(i,j)) || (std::abs(t-src.at<_Tp>(i,j)) < std::numeric_limits<float>::epsilon())) << n;
			}
		}
	}
}


template <typename _Tp> static
inline void calcLBPHistogram_(InputArray _src, OutputArray _dst){
	Mat src = _src.getMat();
	_dst.create(1, 59*4, CV_32FC1);
	Mat dst = _dst.getMat();
	dst.setTo(0);
	Mat temp;
	for(int k=1; k<8; k+=2){
		temp = elbp(src,k,8);
		for(int i = 0; i < temp.rows; i++){
			for(int j = 0; j < temp.cols; j++){
				int bin = UniformPattern59[temp.at<int>(i,j)];
				dst.at<float>(bin+((k-1)/2)*59) += 1;
			}
		}
	}
}

template <typename _Tp> static
inline void calcSIFTDescriptors_(InputArray _src, OutputArray _dst){
	Mat src = _src.getMat();
	_dst.create(1, 128, CV_32FC1);
	Mat dst = _dst.getMat();
	dst.setTo(0);
	
	static VlDsiftFilter* dsift = vl_dsift_new_basic(src.cols, src.rows, src.cols, src.cols/4);
	vl_dsift_set_window_size(dsift, src.cols/2.0);
	
	vector<float> img;
	for(int i = 0; i < src.rows; ++i){
		for(int j = 0; j < src.cols; ++j){
			img.push_back(src.at<_Tp>(i, j));
		}
	}
	
	vl_dsift_process(dsift, &img[0]);
	
	const float* temp =  vl_dsift_get_descriptors(dsift);
	
	for(int i=0; i<128; i++){
		dst.at<float>(i) = temp[i];
	}
	
}



void elbp(InputArray src, OutputArray dst, int radius, int neighbors) {
	switch (src.type()) {
		case CV_8SC1:   elbp_<char>(src,dst, radius, neighbors); break;
		case CV_8UC1:   elbp_<unsigned char>(src, dst, radius, neighbors); break;
		case CV_16SC1:  elbp_<short>(src,dst, radius, neighbors); break;
		case CV_16UC1:  elbp_<unsigned short>(src,dst, radius, neighbors); break;
		case CV_32SC1:  elbp_<int>(src,dst, radius, neighbors); break;
		case CV_32FC1:  elbp_<float>(src,dst, radius, neighbors); break;
		case CV_64FC1:  elbp_<double>(src,dst, radius, neighbors); break;
		default: break;
	}
}

void calcLBPHistogram(InputArray src, OutputArray dst) {
	switch (src.type()) {
		case CV_8SC1:   calcLBPHistogram_<char>(src,dst); break;
		case CV_8UC1:   calcLBPHistogram_<unsigned char>(src, dst); break;
		case CV_16SC1:  calcLBPHistogram_<short>(src,dst); break;
		case CV_16UC1:  calcLBPHistogram_<unsigned short>(src,dst); break;
		case CV_32SC1:  calcLBPHistogram_<int>(src,dst); break;
		case CV_32FC1:  calcLBPHistogram_<float>(src,dst); break;
		case CV_64FC1:  calcLBPHistogram_<double>(src,dst); break;
		default: break;
	}
}

void calcSIFTDescriptors(InputArray src, OutputArray dst) {
	switch (src.type()) {
		case CV_8SC1:   calcSIFTDescriptors_<char>(src,dst); break;
		case CV_8UC1:   calcSIFTDescriptors_<unsigned char>(src, dst); break;
		case CV_16SC1:  calcSIFTDescriptors_<short>(src,dst); break;
		case CV_16UC1:  calcSIFTDescriptors_<unsigned short>(src,dst); break;
		case CV_32SC1:  calcSIFTDescriptors_<int>(src,dst); break;
		case CV_32FC1:  calcSIFTDescriptors_<float>(src,dst); break;
		case CV_64FC1:  calcSIFTDescriptors_<double>(src,dst); break;
		default: break;
	}
}



//------------------------------------------------------------------------------
// elbp
//------------------------------------------------------------------------------
Mat elbp(InputArray src, int radius, int neighbors) {
	Mat dst;
	elbp(src, dst, radius, neighbors);
	return dst;
}

Mat calcLBPHistogram(InputArray src) {
	Mat dst;
	calcLBPHistogram(src, dst);
	return dst;
}

Mat calcSIFTDescriptors(InputArray src) {
	Mat dst;
	calcSIFTDescriptors(src, dst);
	return dst;
}