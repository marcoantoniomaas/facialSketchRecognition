#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include "filters.hpp"
#include "transforms.hpp"
#include "descriptors.hpp"

using namespace std;
using namespace cv;

int main( int argc, char** argv )
{
	Mat src = imread(argv[1],0);
	Mat gabor = magnitude(convolveDFT(src, gaborWavelet(2,3,6.28,25)));
	Mat lbp = elbp(gabor);
	//lbp.convertTo(lbp, CV_32F);
	//Mat dest = radonTransform(lbp);
	imshow("Janela", lbp*256);
	waitKey(0);
	
	return 0;
}

