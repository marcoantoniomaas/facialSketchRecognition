#ifndef __FILTERS_HPP__
#define __FILTERS_HPP__

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

Mat DoGFilter(InputArray src);
Mat GaussianFilter(InputArray src);
Mat CSDNFilter(InputArray src);
Mat gaborWavelet(int mu, int nu, double sigma, int scaleXY);
Mat convolveDFT(InputArray img, InputArray kernel);
Mat magnitude(Mat src);

#endif