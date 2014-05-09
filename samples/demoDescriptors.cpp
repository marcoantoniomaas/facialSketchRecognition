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
	string descriptor = "LRBP";
	string database = "CUFSF";
	
	uint nTraining = 0;
	uint pcaDim = 0;
	uint count = 0;
	
	vector<string> trainingPhotos, trainingSketches, testingPhotos, testingSketches, extraPhotos, photos, sketches;
	
	loadImages(argv[1], photos);
	loadImages(argv[2], sketches);
	
	//auto seed = unsigned(count);
	
	//srand (seed);
	//random_shuffle (sketches.begin(), sketches.end());
	//srand (seed);
	//random_shuffle (photos.begin(), photos.end());
	
	trainingPhotos.insert(trainingPhotos.end(), photos.begin(),photos.begin()+nTraining);
	trainingSketches.insert(trainingSketches.end(), sketches.begin(), sketches.begin()+nTraining);
	testingPhotos.insert(testingPhotos.end(),photos.begin()+nTraining,photos.end());
	testingSketches.insert(testingSketches.end(),sketches.begin()+nTraining,sketches.end());
	
	uint nTestingSketches = testingSketches.size();
	uint nTestingPhotos = testingPhotos.size();
	
	cout << nTestingSketches << " sketches to verify." << endl;
	cout << nTestingPhotos << " photos on the gallery" << endl;
	
	Mat img, temp;
	int size = 32;
	int delta = size/2;
	
	//training 
	vector<int> labels;
	
	for(uint i=0; i<2*nTraining; i++)
		labels.push_back(i%nTraining);
	
	int dim = 0;
	
	if(descriptor=="HOG" || descriptor=="HAOG")
		dim = 9;
	else if(descriptor=="SIFT")
		dim = 128;
	else if(descriptor=="MLBP")
		dim = 236;
	else if(descriptor=="LRBP")
		dim = 32;
	
	Mat data = Mat::zeros(2*nTraining, 154*dim, CV_32F);
	
	PCA pca;
	LDA lda;
	
	#pragma omp parallel for private(img, temp)
	for(uint i=0; i<nTraining; i++){
		img = imread(trainingSketches[i],0);
		
		#pragma omp critical
		temp = extractDescriptors(img, size, delta, filter, descriptor);
		
		temp = temp.t();
		
		temp.copyTo(data.row(i));
	}
	
	#pragma omp parallel for private(img, temp)
	for(uint i=0; i<nTraining; i++){
		img = imread(trainingPhotos[i],0);
		
		#pragma omp critical
		temp = extractDescriptors(img, size, delta, filter, descriptor);
		
		temp = temp.t();
		
		temp.copyTo(data.row(i+nTraining));
	}
	
	if(nTraining>0){
		pca(data, Mat(), CV_PCA_DATA_AS_ROW, pcaDim);
		lda.compute(pca.project(data), labels);
	}
	
	//testing
	vector<Mat*> testingSketchesDescriptors(nTestingSketches), testingPhotosDescriptors(nTestingPhotos);
	
	#pragma omp parallel for private(img, temp)
	for(uint i=0; i<nTestingSketches; i++){
		img = imread(testingSketches[i],0);
		
		testingSketchesDescriptors[i] = new Mat();
		
		#pragma omp critical
		temp = extractDescriptors(img, size, delta, filter, descriptor);
		
		if(nTraining>0)
			*(testingSketchesDescriptors[i]) = lda.project(pca.project(temp.t()));
		else
			*(testingSketchesDescriptors[i]) = temp.clone();
	}
	
	#pragma omp parallel for private(img, temp)
	for(uint i=0; i<nTestingPhotos; i++){
		img = imread(testingPhotos[i],0);
		
		testingPhotosDescriptors[i] = new Mat();
		
		#pragma omp critical
		temp = extractDescriptors(img, size, delta, filter, descriptor);
		
		if(nTraining>0)
			*(testingPhotosDescriptors[i]) = lda.project(pca.project(temp.t()));
		else
			*(testingPhotosDescriptors[i]) = temp.clone();
	}
	
	cerr << "calculating distances" << endl;
	
	Mat distancesChi = Mat::zeros(nTestingSketches,nTestingPhotos,CV_64F);
	Mat distancesL2 = Mat::zeros(nTestingSketches,nTestingPhotos,CV_64F);
	Mat distancesCosine = Mat::zeros(nTestingSketches,nTestingPhotos,CV_64F);
	
	#pragma omp parallel for
	for(uint i=0; i<nTestingSketches; i++){
		for(uint j=0; j<nTestingPhotos; j++){
			distancesChi.at<double>(i,j) = chiSquareDistance(*(testingSketchesDescriptors[i]),*(testingPhotosDescriptors[j]));
		}
	}
	
	string file1name = descriptor + filter + database + to_string(nTraining) + "-" + to_string(pcaDim) + string("chi") + to_string(count) + string(".xml");
	
	FileStorage file1(file1name, FileStorage::WRITE);
	
	file1 << "distanceMatrix" << distancesChi;
	
	file1.release();
	
	return 0;
}