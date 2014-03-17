#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
#include "descriptors.hpp"
#include "utils.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	string filter = "None";
	string descriptor = "HAOG";
	string database = "CUFSF";
	
	int count = 0;
	
	vector<string> testingPhotos, testingSketches, extraPhotos, photos, sketches;
	
	loadImages(argv[1], photos);
	loadImages(argv[2], sketches);
	
	testingPhotos.insert(testingPhotos.end(),photos.begin(),photos.end());
	testingSketches.insert(testingSketches.end(),sketches.begin(),sketches.end());
	
	//testingPhotos.insert(testingPhotos.end(),extraPhotos.begin(),extraPhotos.begin()+10000);
	
	uint nTestingSketches = testingSketches.size();
	uint nTestingPhotos = testingPhotos.size();
	
	cout << nTestingSketches << " sketches to verify." << endl;
	cout << nTestingPhotos << " photos on the gallery" << endl;
	
	Mat img, temp;
	int size = 32;
	int delta = size/2;
	
	//testing
	vector<Mat*> testingSketchesDescriptors(nTestingSketches), testingPhotosDescriptors(nTestingPhotos);
	
	#pragma omp parallel for private(img, temp)
	for(uint i=0; i<nTestingSketches; i++){
		img = imread(testingSketches[i],0);
		//resize(img, img, Size(64,80));
		testingSketchesDescriptors[i] = new Mat();
		
		#pragma omp critical
		temp = extractDescriptors(img, size, delta, filter, descriptor);
		
		*(testingSketchesDescriptors[i]) =temp.clone();
		
		cout << "testingSketches " << i << endl;
	}
	
	#pragma omp parallel for private(img, temp)
	for(uint i=0; i<nTestingPhotos; i++){
		img = imread(testingPhotos[i],0);
		//resize(img, img, Size(64,80));
		testingPhotosDescriptors[i] = new Mat();
		
		#pragma omp critical
		temp = extractDescriptors(img, size, delta, filter, descriptor);
		
		*(testingPhotosDescriptors[i]) = temp.clone();
		
		cout << "testingPhotos " << i << endl;
		//cout << *(testingPhotosDescriptors[i]) << endl;
	}
	
	cerr << "calculating distances" << endl;
	
	
	Mat distancesChi = Mat::zeros(nTestingSketches,nTestingPhotos,CV_64F);
	Mat distancesL2 = Mat::zeros(nTestingSketches,nTestingPhotos,CV_64F);
	Mat distancesCosine = Mat::zeros(nTestingSketches,nTestingPhotos,CV_64F);
	
	#pragma omp parallel for
	for(uint i=0; i<nTestingSketches; i++){
		for(uint j=0; j<nTestingPhotos; j++){
			distancesChi.at<double>(i,j) = chiSquareDistance(*(testingSketchesDescriptors[i]),*(testingPhotosDescriptors[j]));
			distancesL2.at<double>(i,j) = norm(*(testingSketchesDescriptors[i]),*(testingPhotosDescriptors[j]));
			distancesCosine.at<double>(i,j) = abs(1-cosineDistance(*(testingSketchesDescriptors[i]),*(testingPhotosDescriptors[j])));
		}
	}
	
	
	string file1name = descriptor + filter + database + string("chi") + to_string(count) + string(".xml");
	string file2name = descriptor + filter + database + string("l2") + to_string(count) + string(".xml");
	string file3name = descriptor + filter + database + string("cosine") + to_string(count) + string(".xml");
	
	FileStorage file1(file1name, FileStorage::WRITE);
	FileStorage file2(file2name, FileStorage::WRITE);
	FileStorage file3(file3name, FileStorage::WRITE);
	
	file1 << "distanceMatrix" << distancesChi;
	file2 << "distanceMatrix" << distancesL2;
	file3 << "distanceMatrix" << distancesCosine;
	
	file1.release();
	file2.release();
	file3.release();
	
	return 0;
}