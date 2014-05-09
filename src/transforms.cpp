#include "transforms.hpp"

template <typename _Tp> static
inline void radonTransform_(InputArray _src, OutputArray _dst){
	
	//get matrices
	Mat src = _src.getMat();
	
	/// Compute a rotation matrix with respect to the center of the image
	Point center = Point(src.cols/2, src.rows/2);
	double angle = 0.0;
	double scale = 1;
	int diag = ceil(sqrt(src.rows*src.rows+src.cols*src.cols));
	
	// allocate memory for result
	_dst.create(diag, 180, CV_32FC1);
	Mat dst = _dst.getMat();
	// zero
	dst.setTo(0);
	
	Mat rot_mat(2, 3, CV_32FC1);
	Mat rotate_dst;
	
	Size size = Size(diag, diag);
	
	/// Rotate the warped image
	warpAffine(src, rotate_dst, rot_mat, size);
	
	for(angle=0; angle<180; angle++){
		/// Get the rotation matrix with the specifications above
		rot_mat = getRotationMatrix2D( center, 90-angle, scale );
		rot_mat.at<double>(0,2) += (diag - src.cols)/2.0;
		rot_mat.at<double>(1,2) += (diag - src.rows)/2.0;
		
		/// Rotate the warped image
		warpAffine(src, rotate_dst, rot_mat, size);
		
		for(int i=0; i<diag; i++){
			for(int j=0; j<diag; j++){
				dst.at<float>(diag-i-1,angle) += rotate_dst.at<_Tp>(i,j);
			}
		}
	}
	
	normalize(dst, dst, 0, 1, NORM_MINMAX, CV_32FC1);
}

void radonTransform(InputArray src, OutputArray dst) {
	switch (src.type()) {
		case CV_8SC1:   radonTransform_<char>(src, dst); break;
		case CV_8UC1:   radonTransform_<unsigned char>(src, dst); break;
		case CV_16SC1:  radonTransform_<short>(src, dst); break;
		case CV_16UC1:  radonTransform_<unsigned short>(src, dst); break;
		case CV_32SC1:  radonTransform_<int>(src, dst); break;
		case CV_32FC1:  radonTransform_<float>(src, dst); break;
		case CV_64FC1:  radonTransform_<double>(src, dst); break;
		default: break;
	}
}

Mat radonTransform(InputArray src) {
	Mat dst;
	radonTransform(src, dst);
	return dst;
}