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
inline void extractMLBP_(InputArray _src, OutputArray _dst){
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
inline void extractSIFT_(InputArray _src, OutputArray _dst){
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

template <typename _Tp> static
inline void extractHOG_(InputArray _src, OutputArray _dst){
	Mat src = _src.getMat();
	_dst.create(1, 9, CV_32FC1);
	Mat dst = _dst.getMat();
	dst.setTo(0);

	Mat grad_x, grad_y, magn, theta;
	
	Sobel(src, grad_x, CV_32F, 1, 0);
	Sobel(src, grad_y, CV_32F, 0, 1);
	
	magn = Mat::zeros(src.size(), CV_32F);
	theta = Mat::zeros(src.size(), CV_32F);
	
	for(int x=0; x<src.rows; x++){
		for(int y=0; y<src.cols; y++){
			magn.at<float>(x,y) = sqrt(pow(grad_x.at<float>(x,y),2)+pow(grad_y.at<float>(x,y),2));
			theta.at<float>(x,y) = atan2(grad_y.at<float>(x,y),grad_x.at<float>(x,y))* 180 / CV_PI;;
		}
	}
	
	for(int x=0; x<src.rows; x++){
		for(int y=0; y<src.cols; y++){
			int index = ceil((theta.at<float>(x,y)+180)/40)-1;
			dst.at<float>(index)+=magn.at<float>(x,y);
		}
	}
}

template <typename _Tp> static
inline void extractHAOG_(InputArray _src, OutputArray _dst){
	Mat src = _src.getMat();
	_dst.create(1, 9, CV_32FC1);
	Mat dst = _dst.getMat();
	dst.setTo(0);

	Mat grad_x, grad_y, magn, theta;
	
	Sobel(src, grad_x, CV_32F, 1, 0);
	Sobel(src, grad_y, CV_32F, 0, 1);
	
	magn = Mat::zeros(src.size(), CV_32F);
	theta = Mat::zeros(src.size(), CV_32F);
	
	for(int x=0; x<src.rows; x++){
		for(int y=0; y<src.cols; y++){
			magn.at<float>(x,y) = sqrt(pow(grad_x.at<float>(x,y),2)+pow(grad_y.at<float>(x,y),2));
		}
	}
	
	Mat grad_sx = Mat::zeros(src.size(), CV_32F), 
	grad_sy = Mat::zeros(src.size(), CV_32F),
	magn_sq = Mat::zeros(src.size(), CV_32F);
	
	
	for(int x=0; x<src.rows; x++){
		for(int y=0; y<src.cols; y++){
			int x0 = x-1>=0? x-1: 0;
			int x1 = x+1< src.rows ? x+1: src.rows-1;
			int y0 = y-1>=0? y-1: 0;
			int y1 = y+1< src.cols ? y+1: src.cols-1;
			for(int i=x0; i<=x1; i++){
				for(int j=y0; j<y1; j++){
					grad_sx.at<float>(x,y)+= pow(grad_x.at<float>(i,j),2)-pow(grad_y.at<float>(i,j),2);
					grad_sy.at<float>(x,y)+= 2*grad_x.at<float>(i,j)*grad_y.at<float>(i,j);
					magn_sq.at<float>(x,y)+= pow(magn.at<float>(i,j),2);
				}
			}
		}
	}
	
	for(int x=0; x<src.rows; x++){
		for(int y=0; y<src.cols; y++){
			theta.at<float>(x,y) = atan2(grad_sy.at<float>(x,y),grad_sx.at<float>(x,y))* 180 / CV_PI;
		}
	}
	
	for(int x=0; x<src.rows; x++){
		for(int y=0; y<src.cols; y++){
			int index = ceil((theta.at<float>(x,y)+180)/40)-1;
			dst.at<float>(index)+=magn_sq.at<float>(x,y);
		}
	}
}

template <typename _Tp> static
inline void extractLRBP_(InputArray _src, OutputArray _dst){
	
	/// Establish the number of bins
	int histSize = 32;
	/// Set the ranges of histogram
	float range[] = {0, 255} ;
	const float* histRange = { range };
	
	Mat src = _src.getMat();
	_dst.create(1, histSize, CV_32FC1);
	Mat dst = _dst.getMat();
	dst.setTo(0);
	
	Mat radon = radonTransform(src);
	Mat lrbp = elbp(radon, 2, 8);
	lrbp.convertTo(lrbp, CV_32F);
	Mat hist;
	calcHist(&lrbp, 1, 0, Mat(), hist, 1, &histSize, &histRange);
	
	for(int i=0; i<histSize; i++){
		dst.at<float>(i)=hist.at<float>(i);
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

void extractMLBP(InputArray src, OutputArray dst) {
	switch (src.type()) {
		case CV_8SC1:   extractMLBP_<char>(src,dst); break;
		case CV_8UC1:   extractMLBP_<unsigned char>(src, dst); break;
		case CV_16SC1:  extractMLBP_<short>(src,dst); break;
		case CV_16UC1:  extractMLBP_<unsigned short>(src,dst); break;
		case CV_32SC1:  extractMLBP_<int>(src,dst); break;
		case CV_32FC1:  extractMLBP_<float>(src,dst); break;
		case CV_64FC1:  extractMLBP_<double>(src,dst); break;
		default: break;
	}
}

void extractSIFT(InputArray src, OutputArray dst) {
	switch (src.type()) {
		case CV_8SC1:   extractSIFT_<char>(src,dst); break;
		case CV_8UC1:   extractSIFT_<unsigned char>(src, dst); break;
		case CV_16SC1:  extractSIFT_<short>(src,dst); break;
		case CV_16UC1:  extractSIFT_<unsigned short>(src,dst); break;
		case CV_32SC1:  extractSIFT_<int>(src,dst); break;
		case CV_32FC1:  extractSIFT_<float>(src,dst); break;
		case CV_64FC1:  extractSIFT_<double>(src,dst); break;
		default: break;
	}
}

void extractHOG(InputArray src, OutputArray dst) {
	switch (src.type()) {
		case CV_8SC1:   extractHOG_<char>(src,dst); break;
		case CV_8UC1:   extractHOG_<unsigned char>(src, dst); break;
		case CV_16SC1:  extractHOG_<short>(src,dst); break;
		case CV_16UC1:  extractHOG_<unsigned short>(src,dst); break;
		case CV_32SC1:  extractHOG_<int>(src,dst); break;
		case CV_32FC1:  extractHOG_<float>(src,dst); break;
		case CV_64FC1:  extractHOG_<double>(src,dst); break;
		default: break;
	}
}

void extractHAOG(InputArray src, OutputArray dst) {
	switch (src.type()) {
		case CV_8SC1:   extractHAOG_<char>(src,dst); break;
		case CV_8UC1:   extractHAOG_<unsigned char>(src, dst); break;
		case CV_16SC1:  extractHAOG_<short>(src,dst); break;
		case CV_16UC1:  extractHAOG_<unsigned short>(src,dst); break;
		case CV_32SC1:  extractHAOG_<int>(src,dst); break;
		case CV_32FC1:  extractHAOG_<float>(src,dst); break;
		case CV_64FC1:  extractHAOG_<double>(src,dst); break;
		default: break;
	}
}

void extractLRBP(InputArray src, OutputArray dst) {
	switch (src.type()) {
		case CV_8SC1:   extractLRBP_<char>(src,dst); break;
		case CV_8UC1:   extractLRBP_<unsigned char>(src, dst); break;
		case CV_16SC1:  extractLRBP_<short>(src,dst); break;
		case CV_16UC1:  extractLRBP_<unsigned short>(src,dst); break;
		case CV_32SC1:  extractLRBP_<int>(src,dst); break;
		case CV_32FC1:  extractLRBP_<float>(src,dst); break;
		case CV_64FC1:  extractLRBP_<double>(src,dst); break;
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

Mat extractMLBP(InputArray src) {
	Mat dst;
	extractMLBP(src, dst);
	return dst;
}

Mat extractSIFT(InputArray src) {
	Mat dst;
	extractSIFT(src, dst);
	return dst;
}

Mat extractHOG(InputArray src){
	Mat dst;
	extractHOG(src, dst);
	return dst;
}

Mat extractHAOG(InputArray src){
	Mat dst;
	extractHAOG(src, dst);
	return dst;
}

Mat extractLRBP(InputArray src){
	Mat dst;
	extractLRBP(src, dst);
	return dst;
}
