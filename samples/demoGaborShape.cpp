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
	Mat dest = radonTransform(src);
	Mat lbp = elbp(dest);
	imshow("Janela", lbp*256);
	waitKey(0);
	
	return 0;
}

