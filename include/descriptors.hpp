#ifndef __DESCRIPTORS_HPP__
#define __DESCRIPTORS_HPP__

#include <iostream>
#include <limits>
#include <opencv2/core/core.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <vl/dsift.h>
#include <vl/generic.h>
#include "transforms.hpp"

using namespace cv;
//using namespace std;

void elbp(InputArray src, OutputArray dst, int radius=1, int neighbors=8);
Mat elbp(InputArray src, int radius=1, int neighbors=8);
void extractMLBP(InputArray src, OutputArray dst);
Mat extractMLBP(InputArray src);
void extractSIFT(InputArray src, OutputArray dst);
Mat extractSIFT(InputArray src);
void extractHOG(InputArray src, OutputArray dst);
Mat extractHOG(InputArray src);
void extractHAOG(InputArray src, OutputArray dst);
Mat extractHAOG(InputArray src);
void extractLRBP(InputArray src, OutputArray dst);
Mat extractLRBP(InputArray src);

#endif