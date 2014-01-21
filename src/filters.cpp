#include "filters.hpp"

Mat DoGFilter(InputArray src){
	
	Mat g1, g2, dst;
	GaussianBlur(src, g1, Size(5,5), 0);
	GaussianBlur(src, g2, Size(9,9), 0);
	
	dst = g2 - g1;
	normalize(dst,dst,0, 255, NORM_MINMAX, CV_8U);
	
	return dst;
}

Mat GaussianFilter(InputArray src){
	
	Mat dst;
	GaussianBlur(src, dst, Size(5,5), 0);
	
	return dst;
}

Mat CSDNFilter(InputArray _src){
	
	Mat dst;
	Mat src = _src.getMat();
	src.convertTo(src, CV_32F);
	
	blur(src, dst, Size(16,16));
	divide(src, dst, dst);
	normalize(dst, dst, 0, 255, NORM_MINMAX, CV_8U);
	
	return dst; 
}

Mat gaborWavelet(int mu, int nu, double sigma, int scaleXY){
	
	double kmax = CV_PI/2;
	double f = sqrt(2);
	double angStep = CV_PI/8;
	
	if(scaleXY == 0){
		double th = 5e-3;
		scaleXY = ceil(sqrt(-log(th*pow(sigma,2)/pow(kmax,2))*2*pow(sigma,2)/pow(kmax,2)));
	}
	
	double DC = exp(-pow(sigma,2)/2);
	
	Mat G = Mat::zeros(2*scaleXY+1,2*scaleXY+1,CV_32FC2);
	for(int x=-scaleXY; x<= scaleXY; x++){
		for(int y=-scaleXY; y<=scaleXY; y++){
			double phi = angStep*mu;
			double k = kmax/pow(f,nu);
			G.at<complex< float >>(y+scaleXY,x+scaleXY) = pow(k,2)/pow(sigma,2)*exp(-pow(k,2)*(pow(x,2)+pow(y,2))/2/pow(sigma,2))*
			(exp(complex< double >(0,1)*(k*cos(phi)*x+k*sin(phi)*y)) - DC);
		}
	}
	
	return G;
}

Mat convolveDFT(InputArray img, InputArray kernel){
	
	//get matrices
	Mat A = img.getMat();
	Mat B = kernel.getMat();
	
	if(!(A.type() == CV_32FC1 || A.type() == CV_32FC2 || A.type() == CV_64FC1 || A.type() == CV_64FC2))
		A.convertTo(A, CV_32F);
	
	if(!(B.type() == CV_32FC1 || B.type() == CV_32FC2 || B.type() == CV_64FC1 || B.type() == CV_64FC2))
		B.convertTo(B, CV_32F);
	
	
	Mat C;
	// reallocate the output array if needed
	//C.create(abs(A.rows - B.rows)+1, abs(A.cols - B.cols)+1, A.type());
	C.create(A.rows, A.cols, A.type());
	Size dftSize;
	// calculate the size of DFT transform
	dftSize.width = getOptimalDFTSize(A.cols + B.cols - 1);
	dftSize.height = getOptimalDFTSize(A.rows + B.rows - 1);
	
	// allocate temporary buffers and initialize them with 0's
	Mat tempA(dftSize, A.type(), Scalar::all(0));
	Mat tempB(dftSize, B.type(), Scalar::all(0));
	
	// copy A and B to the top-left corners of tempA and tempB, respectively
	Mat roiA(tempA, Rect(0,0,A.cols,A.rows));
	A.copyTo(roiA);
	Mat roiB(tempB, Rect(0,0,B.cols,B.rows));
	B.copyTo(roiB);
	
	// now transform the padded A & B in-place;
	// use "nonzeroRows" hint for faster processing
	dft(tempA, tempA, DFT_COMPLEX_OUTPUT, A.rows);
	dft(tempB, tempB, DFT_COMPLEX_OUTPUT, B.rows);
	
	// multiply the spectrums;
	// the function handles packed spectrum representations well
	mulSpectrums(tempA, tempB, tempA,0);
	
	// transform the product back from the frequency domain.
	// Even though all the result rows will be non-zero,
	// you need only the first C.rows of them, and thus you
	// pass nonzeroRows == C.rows
	dft(tempA, tempA, DFT_INVERSE + DFT_SCALE, C.rows);
	
	// now copy the result back to C.
	tempA(Rect(0, 0, C.cols, C.rows)).copyTo(C);
	
	return C;
	// all the temporary buffers will be deallocated automatically
}

Mat magnitude(Mat src){
	
	CV_Assert(src.type() == CV_32FC2 || src.type() == CV_64FC2);
	
	Mat dst;
	vector<Mat> planes;
	split(src, planes);
	magnitude(planes[0], planes[1], dst);
	normalize(dst, dst, 0, 1, NORM_MINMAX);
	
	return dst;
}