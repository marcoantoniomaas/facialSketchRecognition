#ifndef __DESCRIPTORS_HPP__
#define __DESCRIPTORS_HPP__

#include <iostream>
#include <limits>
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <vl/dsift.h>
#include <vl/generic.h>

using namespace cv;
//using namespace std;

void elbp(InputArray src, OutputArray dst, int radius=1, int neighbors=8);
Mat elbp(InputArray src, int radius=1, int neighbors=8);
void calcLBPHistogram(InputArray src, OutputArray dst);
Mat calcLBPHistogram(InputArray src);
void calcSIFTDescriptors(InputArray src, OutputArray dst);
Mat calcSIFTDescriptors(InputArray src);

#endif