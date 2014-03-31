#include "kernelproto.hpp"

Kernel::Kernel(vector<Mat*> &trainingPhotosDescriptors, vector<Mat*> &trainingSketchesDescriptors)
{
	this->trainingPhotosDescriptors = trainingPhotosDescriptors;
	this->trainingSketchesDescriptors = trainingSketchesDescriptors;
}

Kernel::~Kernel()
{
	
}

void Kernel::compute()
{
	
	int n = trainingPhotosDescriptors.size();
	
	Kp = Mat::zeros(n,n,CV_32F);
	Kg = Mat::zeros(n,n,CV_32F);
	
	for(int i=0; i<n; i++)
		for(int j=0; j<n; j++){
			Kg.at<float>(i,j) = cosineDistance(*trainingPhotosDescriptors[i], *trainingPhotosDescriptors[j]);
			Kp.at<float>(i,j) = cosineDistance(*trainingSketchesDescriptors[i],*trainingSketchesDescriptors[j]);
		}
		
		R = Kg*((Kp).t()*Kp).inv()*(Kp).t();
}

Mat Kernel::projectGallery(Mat desc)
{
	int n = trainingPhotosDescriptors.size();
	Mat result = Mat::zeros(1,n,CV_32F);
	for(int i=0; i<n; i++)
		result.at<float>(i) = cosineDistance(desc,*trainingPhotosDescriptors[i]);
	
	//normalize(result,result,1,0,NORM_MINMAX, CV_32F);
	
	return result.t();
}

Mat Kernel::projectProbe(Mat desc)
{
	int n = trainingSketchesDescriptors.size();
	Mat result = Mat::zeros(1,n,CV_32F);
	for(int i=0; i<n; i++)
		result.at<float>(i) = cosineDistance(desc,*trainingSketchesDescriptors[i]);
	
	result = R*result.t();
	//normalize(result,result,1,0,NORM_MINMAX, CV_32F);
	
	return result;
}