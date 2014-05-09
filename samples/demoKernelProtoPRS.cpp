#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
#include "descriptors.hpp"
#include "utils.hpp"
#include "filters.hpp"
#include "kernelproto.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	string filter = "Gaussian";
	string descriptor = "SIFT";
	string database = "Forensic-extra";
	uint count = 1;
	
	vector<string> extraPhotos, photos, sketches;
	
	loadImages(argv[5], photos);
	loadImages(argv[6], sketches);
	loadImages(argv[7], extraPhotos);
	
	uint nPhotos = photos.size(),
	nSketches = sketches.size(),
	nExtra = extraPhotos.size();
	
	uint nTraining = 2*nPhotos/3;
	
	cout << "Read " << nSketches << " sketches." << endl;
	cout << "Read " << nPhotos + nExtra << " photos." << endl;
	
	vector<Mat*> sketchesDescriptors(nSketches), photosDescriptors(nPhotos), extraDescriptors(nExtra);
	
	Mat img, temp;
	
	int size=32, delta=16;
	
	#pragma omp parallel for private(img, temp)
	for(uint i=0; i<nSketches; i++){
		img = imread(sketches[i],0);
		sketchesDescriptors[i] = new Mat();
		
		#pragma omp critical
		temp = extractDescriptors(img, size, delta, filter, descriptor);
		
		*(sketchesDescriptors[i]) = temp.clone();
	}
	
	#pragma omp parallel for private(img, temp)
	for(uint i=0; i<nPhotos; i++){
		img = imread(photos[i],0);
		photosDescriptors[i] = new Mat();
		
		#pragma omp critical
		temp = extractDescriptors(img, size, delta, filter, descriptor);
		
		*(photosDescriptors[i]) = temp.clone();
	}
	
	#pragma omp parallel for private(img, temp)
	for(uint i=0; i<nExtra; i++){
		img = imread(extraPhotos[i],0);
		extraDescriptors[i] = new Mat();
		
		#pragma omp critical
		temp = extractDescriptors(img, size, delta, filter, descriptor);
		
		*(extraDescriptors[i]) = temp.clone();
	}
	
	auto seed = unsigned(count);
	
	srand(seed);
	random_shuffle(sketchesDescriptors.begin(), sketchesDescriptors.end());
	srand(seed);
	random_shuffle(photosDescriptors.begin(), photosDescriptors.end());
	
	//training
	vector<Mat*> trainingSketchesDescriptors1, trainingPhotosDescriptors1, 
	trainingSketchesDescriptors2, trainingPhotosDescriptors2;
	
	trainingSketchesDescriptors1.insert(trainingSketchesDescriptors1.end(), sketchesDescriptors.begin(), sketchesDescriptors.begin()+nTraining/2);
	trainingPhotosDescriptors1.insert(trainingPhotosDescriptors1.end(), photosDescriptors.begin(), photosDescriptors.begin()+nTraining/2);
	trainingSketchesDescriptors2.insert(trainingSketchesDescriptors2.end(), sketchesDescriptors.begin()+nTraining/2, sketchesDescriptors.begin()+nTraining);
	trainingPhotosDescriptors2.insert(trainingPhotosDescriptors2.end(), photosDescriptors.begin()+nTraining/2, photosDescriptors.begin()+nTraining);
	
	uint nTraining1 = trainingPhotosDescriptors1.size(),
	nTraining2 = trainingPhotosDescriptors2.size();
	
	//testing
	vector<Mat*> testingSketchesDescriptors, testingPhotosDescriptors;
	
	testingSketchesDescriptors.insert(testingSketchesDescriptors.end(), sketchesDescriptors.begin()+nTraining, sketchesDescriptors.end());
	testingPhotosDescriptors.insert(testingPhotosDescriptors.end(), photosDescriptors.begin()+nTraining, photosDescriptors.end());
	testingPhotosDescriptors.insert(testingPhotosDescriptors.end(), extraDescriptors.begin(), extraDescriptors.end());
	
	uint nTestingSketches = testingSketchesDescriptors.size(),
	nTestingPhotos = testingPhotosDescriptors.size();
	
	PCA pca;
	LDA lda;
	vector<int> labels;
	
	for(uint i=0; i<nTraining2; i++){
		labels.push_back(i);
	}
	labels.insert(labels.end(),labels.begin(),labels.end());
	
	//bags
	vector<Mat*> testingSketchesDescriptorsBag(nTestingSketches), testingPhotosDescriptorsBag(nTestingPhotos), 
	trainingPhotosDescriptors1Temp(nTraining1), trainingSketchesDescriptors1Temp(nTraining1);
	
	for(int b=0; b<30; b++){
		
		vector<int> bag_indexes = gen_bag(154, 0.1);
		
		#pragma omp parallel for private(temp)
		for(uint i=0; i<nTraining1; i++){
			temp = *(trainingSketchesDescriptors1[i]);
			temp = bag(temp, bag_indexes, 154);
			trainingSketchesDescriptors1Temp[i] = new Mat();
			*(trainingSketchesDescriptors1Temp[i]) = temp.clone();
		}
		
		#pragma omp parallel for private(temp)
		for(uint i=0; i<nTraining1; i++){
			temp = *(trainingPhotosDescriptors1[i]);
			temp = bag(temp, bag_indexes, 154);
			trainingPhotosDescriptors1Temp[i] = new Mat();
			*(trainingPhotosDescriptors1Temp[i]) = temp.clone();
		}
		
		Kernel k(trainingPhotosDescriptors1Temp, trainingSketchesDescriptors1Temp);
		k.compute();
		
		uint dim = (k.projectGallery(bag(*(trainingPhotosDescriptors1[0]), bag_indexes, 154))).total();
		
		Mat X(dim, 2*nTraining2, CV_32F);
		
		#pragma omp parallel for private(temp)
		for(uint i=0; i<nTraining2; i++){
			temp = *(trainingSketchesDescriptors2[i]);
			temp = bag(temp, bag_indexes, 154);
			temp = k.projectProbe(temp);
			temp.copyTo(X.col(i));
		}
		
		#pragma omp parallel for private(temp)
		for(uint i=0; i<nTraining2; i++){
			temp = *(trainingPhotosDescriptors2[i]);
			temp = bag(temp, bag_indexes, 154);
			temp = k.projectGallery(temp);
			temp.copyTo(X.col(i+nTraining2));
		}
		
		Mat meanX = Mat::zeros(dim, 1, CV_32F), instance;
		
		// calculate sums
		for (int i = 0; i < X.cols; i++) {
			instance = X.col(i);
			add(meanX, instance, meanX);
		}
		
		// calculate total mean
		meanX.convertTo(meanX, CV_32F, 1.0/static_cast<double>(X.cols));
		
		// subtract the mean of matrix
		for(int i=0; i<X.cols; i++) {
			Mat c_i = X.col(i);
			subtract(c_i, meanX.reshape(1,dim), c_i);
		}
		
		pca.computeVar(X, Mat(), CV_PCA_DATA_AS_COL, .99);
		
		Mat W1 = pca.eigenvectors.t();
		Mat ldaData = (W1.t()*X).t();
		lda.compute(ldaData, labels);
		Mat W2 = lda.eigenvectors();
		W2.convertTo(W2, CV_32F);
		Mat projectionMatrix = (W2.t()*W1.t()).t();
		
		//testing
		#pragma omp parallel for private(temp)
		for(uint i=0; i<nTestingSketches; i++){
			temp = *(testingSketchesDescriptors[i]);
			temp = bag(temp, bag_indexes, 154);
			temp = k.projectProbe(temp);
			temp = projectionMatrix.t()*(temp-meanX);
			if(b==0){
				testingSketchesDescriptorsBag[i] = new Mat();
				*(testingSketchesDescriptorsBag[i]) = temp.clone();
			}
			else{
				vconcat(*(testingSketchesDescriptorsBag[i]), temp, *(testingSketchesDescriptorsBag[i]));
			}
		}
		
		#pragma omp parallel for private(temp)
		for(uint i=0; i<nTestingPhotos; i++){
			temp = *(testingPhotosDescriptors[i]);
			temp = bag(temp, bag_indexes, 154);
			temp = k.projectGallery(temp);
			temp = projectionMatrix.t()*(temp-meanX);
			if(b==0){
				testingPhotosDescriptorsBag[i] = new Mat();
				*(testingPhotosDescriptorsBag[i]) = temp.clone();
			}
			else{
				vconcat(*(testingPhotosDescriptorsBag[i]), temp, *(testingPhotosDescriptorsBag[i]));
			}
		}
	}
	
	Mat distancesCosine = Mat::zeros(nTestingSketches,nTestingPhotos,CV_64F);
	
	#pragma omp parallel for
	for(uint i=0; i<nTestingSketches; i++){
		for(uint j=0; j<nTestingPhotos; j++){
			distancesCosine.at<double>(i,j) = abs(1-cosineDistance(*(testingSketchesDescriptorsBag[i]),*(testingPhotosDescriptorsBag[j])));
		}
	}
	
	string file1name = "kernel-prs-" + filter + descriptor + database + to_string(nTraining) + string("cosine") + to_string(count) + string(".xml");
	
	FileStorage file1(file1name, FileStorage::WRITE);
	
	file1 << "distanceMatrix" << distancesCosine;
	
	file1.release();
	
	return 0;
}