#ifndef __TRANSFORMS_HPP__
#define __TRANSFORMS_HPP__

#include <opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>

using namespace cv;

void radonTransform(InputArray src, OutputArray dst);
Mat radonTransform(InputArray src);

#endif