#ifndef __KERNELPROTO_HPP__
#define __KERNELPROTO_HPP__

#include <vector>
#include <opencv2/core/core.hpp>
#include <string>
#include "utils.hpp"

using namespace std;
using namespace cv;

class Kernel
{
private:
	vector<Mat*> trainingPhotosDescriptors, trainingSketchesDescriptors;
	Mat Kp, Kg, R, mean;
public:
	Kernel(vector<Mat*>& trainingPhotosDescriptors,vector<Mat*>& trainingSketchesDescriptors);
	virtual ~Kernel();
	void compute();
	Mat projectGallery(Mat descriptor);
	Mat projectProbe(Mat descriptor);
};

#endif