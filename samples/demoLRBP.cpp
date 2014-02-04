#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include "filters.hpp"
#include "transforms.hpp"
#include "descriptors.hpp"
#include "utils.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	
	vector<string> trainingPhotos, trainingSketches, testingPhotos, testingSketches, extraPhotos;
	
	//loadImages(argv[1], trainingPhotos);
	//loadImages(argv[2], trainingSketches);
	loadImages(argv[1], testingPhotos);
	loadImages(argv[2], testingSketches);
	//loadImages(argv[5], extraPhotos);
	
	
	
	uint nTestingSketches = testingSketches.size();
	uint nTestingPhotos = testingPhotos.size() + extraPhotos.size();
	//testing
	
	
	cerr << "calculating distances" << endl;
	
	Mat distances = Mat::zeros(nTestingSketches,nTestingPhotos,CV_64F);
	FileStorage file("lrbp.xml", FileStorage::WRITE);
	
	#pragma omp parallel for
	for(uint i=0; i<nTestingSketches; i++){
		for(uint j=0; j<nTestingPhotos; j++){
			distances.at<double>(i,j) = 0;//norm();//chiSquareDistance();
		}
	}
	
	file << "distanceMatrix" << distances;
	file.release();
	
	return 0;
}