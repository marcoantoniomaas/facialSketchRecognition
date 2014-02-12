#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include "filters.hpp"
#include "transforms.hpp"
#include "descriptors.hpp"
#include "utils.hpp"

using namespace std;
using namespace cv;


Mat extractHAOG(InputArray _img){
	
	Mat img = _img.getMat();
	int scale = 1;
	int delta = 0;
	int ddepth = CV_32F;
	
	Mat magn, ori;
	
	/// Generate grad_x and grad_y
	Mat grad_x, grad_y;
	
	/// Gradient X
	//Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
	Sobel( img, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
	
	/// Gradient Y
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel( img, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
	
	magnitude(grad_x, grad_y, magn);
	magn = magn/256;
	magn.convertTo(magn, CV_32F);
	
	ori = Mat::zeros(grad_x.size(), CV_32F);
	
	for(int x=0; x<ori.rows; x++){
		for(int y=0; y<ori.cols; y++){
			ori.at<float>(x,y) = atan(grad_y.at<float>(x,y)/grad_x.at<float>(x,y));
		}
	}
	
	normalize(ori, ori, 1, 0, NORM_MINMAX);
	return ori;
}

int main(int argc, char** argv)
{
	Mat img = imread("/home/marco/workspace/database/photos/9.png",0);
	Mat result = extractHAOG(img);
	
	imshow("Janela 1", result);
	waitKey(0);
	
	return 0;
}