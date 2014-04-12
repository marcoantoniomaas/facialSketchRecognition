#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
#include "descriptors.hpp"
#include "utils.hpp"
#include "filters.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	
	string filter = "Gaussian";
	string descriptor = "HAOG";
	string database = "CUFSF";
	int count = 0;
	
	vector<string> extraPhotos, photos, sketches;
	
	loadImages(argv[1], photos);
	loadImages(argv[2], sketches);
	//loadImages(argv[7], extraPhotos);
	
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
	
	auto seed = unsigned(0);
	
	srand(seed);
	random_shuffle(sketches.begin(), sketches.end());
	srand(seed);
	random_shuffle(photos.begin(), photos.end());
	
	//training
	vector<Mat*> trainingSketchesDescriptors, trainingPhotosDescriptors;
	
	trainingSketchesDescriptors.insert(trainingSketchesDescriptors.end(), sketchesDescriptors.begin(), sketchesDescriptors.begin()+nTraining);
	trainingPhotosDescriptors.insert(trainingPhotosDescriptors.end(), photosDescriptors.begin(), photosDescriptors.begin()+nTraining);
	
	//testing
	vector<Mat*> testingSketchesDescriptors, testingPhotosDescriptors;
	
	testingSketchesDescriptors.insert(testingSketchesDescriptors.end(), sketchesDescriptors.begin()+nTraining, sketchesDescriptors.end());
	testingPhotosDescriptors.insert(testingPhotosDescriptors.end(), photosDescriptors.begin()+nTraining, photosDescriptors.end());
	testingPhotosDescriptors.insert(testingPhotosDescriptors.end(), extraDescriptors.begin(), extraDescriptors.end());
	
	PCA pca;
	LDA lda;
	vector<int> labels;
	
	uint nTestingSketches = testingSketchesDescriptors.size(),
	nTestingPhotos = testingPhotosDescriptors.size();
	
	for(uint i=0; i<nTraining; i++){
		labels.push_back(i);
	}
	labels.insert(labels.end(),labels.begin(),labels.end());
	
	//bags
	vector<Mat*> testingSketchesDescriptorsBag(nTestingSketches), testingPhotosDescriptorsBag(nTestingPhotos);
	
	for(int b=0; b<200; b++){
		
		vector<int> bag_indexes = gen_bag(154, 0.1);
		
		uint dim = (bag(*(trainingPhotosDescriptors[0]), bag_indexes, 154)).total();
		
		Mat X(dim, 2*nTraining, CV_32F);
		
		#pragma omp parallel for private(img, temp)
		for(uint i=0; i<nTraining; i++){
			temp = *(trainingSketchesDescriptors[i]);
			temp = bag(temp, bag_indexes, 154);
			temp.copyTo(X.col(i));
		}
		
		#pragma omp parallel for private(img, temp)
		for(uint i=0; i<nTraining; i++){
			temp = *(trainingPhotosDescriptors[i]);
			temp = bag(temp, bag_indexes, 154);
			temp.copyTo(X.col(i+nTraining));
		}
		
		Mat Xs = X(Range::all(), Range(0,nTraining));
		Mat Xp = X(Range::all(), Range(nTraining,2*nTraining));
		
		Mat meanX = Mat::zeros(dim, 1, CV_32F), instance;
		Mat meanXs = Mat::zeros(dim, 1, CV_32F);
		Mat meanXp = Mat::zeros(dim, 1, CV_32F);
		
		// calculate sums
		for (int i = 0; i < X.cols; i++) {
			instance = X.col(i);
			add(meanX, instance, meanX);
		}
		
		for (int i = 0; i < Xs.cols; i++) {
			instance = Xs.col(i);
			add(meanXs, instance, meanXs);
		}
		
		for (int i = 0; i < Xp.cols; i++) {
			instance = Xp.col(i);
			add(meanXp, instance, meanXp);
		}
		
		// calculate total mean
		meanX.convertTo(meanX, CV_32F, 1.0/static_cast<double>(X.cols));
		meanXs.convertTo(meanXs, CV_32F, 1.0/static_cast<double>(Xs.cols));
		meanXp.convertTo(meanXp, CV_32F, 1.0/static_cast<double>(Xp.cols));
		
		
		// subtract the mean of matrix
		for(int i=0; i<X.cols; i++) {
			Mat c_i = X.col(i);
			subtract(c_i, meanX.reshape(1,dim), c_i);
		}
		
		for(int i=0; i<Xs.cols; i++) {
			Mat c_i = Xs.col(i);
			subtract(c_i, meanXs.reshape(1,dim), c_i);
		}
		
		for(int i=0; i<Xp.cols; i++) {
			Mat c_i = Xp.col(i);
			subtract(c_i, meanXp.reshape(1,dim), c_i);
		}
		
		if(meanX.total() >= nTraining)
			pca(X, Mat(), CV_PCA_DATA_AS_COL, nTraining-1);
		else
			pca.computeVar(X, Mat(), CV_PCA_DATA_AS_COL, .99);
		
		Mat W1 = pca.eigenvectors.t();
		Mat ldaData = (W1.t()*X).t();
		lda.compute(ldaData, labels);
		Mat W2 = lda.eigenvectors();
		W2.convertTo(W2, CV_32F);
		Mat projectionMatrix = (W2.t()*W1.t()).t();
		
		//testing
		#pragma omp parallel for private(img, temp)
		for(uint i=0; i<nTestingSketches; i++){
			temp = *(testingSketchesDescriptors[i]);
			temp = bag(temp, bag_indexes, 154);
			temp = projectionMatrix.t()*(temp-meanX);
			if(b==0){
				testingSketchesDescriptorsBag[i] = new Mat();
				*(testingSketchesDescriptorsBag[i]) = temp.clone();
			}
			else{
				vconcat(*(testingSketchesDescriptorsBag[i]), temp, *(testingSketchesDescriptorsBag[i]));
			}
		}
		
		#pragma omp parallel for private(img, temp)
		for(uint i=0; i<nTestingPhotos; i++){
			temp = *(testingPhotosDescriptors[i]);
			temp = bag(temp, bag_indexes, 154);
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
	
	Mat distancesChi = Mat::zeros(nTestingSketches,nTestingPhotos,CV_64F);
	Mat distancesL2 = Mat::zeros(nTestingSketches,nTestingPhotos,CV_64F);
	Mat distancesCosine = Mat::zeros(nTestingSketches,nTestingPhotos,CV_64F);
	
	#pragma omp parallel for
	for(uint i=0; i<nTestingSketches; i++){
		for(uint j=0; j<nTestingPhotos; j++){
			distancesChi.at<double>(i,j) = chiSquareDistance(*(testingSketchesDescriptorsBag[i]),*(testingPhotosDescriptorsBag[j]));
			distancesL2.at<double>(i,j) = norm(*(testingSketchesDescriptorsBag[i]),*(testingPhotosDescriptorsBag[j]));
			distancesCosine.at<double>(i,j) = abs(1-cosineDistance(*(testingSketchesDescriptorsBag[i]),*(testingPhotosDescriptorsBag[j])));
		}
	}
	
	string file1name = "kernel-drs-" + descriptor + database + to_string(nTraining) + string("chi") + to_string(count) + string(".xml");
	string file2name = "kernel-drs-" + descriptor + database + to_string(nTraining) + string("l2") + to_string(count) + string(".xml");
	string file3name = "kernel-drs-" + descriptor + database + to_string(nTraining) + string("cosine") + to_string(count) + string(".xml");
	
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