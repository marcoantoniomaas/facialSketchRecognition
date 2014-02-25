#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
#include "descriptors.hpp"
#include "utils.hpp"
#include "filters.hpp"

using namespace std;
using namespace cv;

Mat extractDescriptors(InputArray src, int size, int delta){
	
	Mat img = src.getMat();
	img = DoGFilter(img);
	//img = GaussianFilter(img);
	//img = CSDNFilter(img);
	int w = img.cols, h=img.rows;
	int n = (w-size)/delta+1, m=(h-size)/delta+1;
	int point = 0;
	
	Mat result = Mat::zeros(m*n*236, 1, CV_32F);
	Mat desc, temp;
	
	for(int i=0;i<=w-size;i+=(size-delta)){
		for(int j=0; j<=h-size; j+=(size-delta)){
			temp = img(Rect(i,j,size,size));
			//extractHAOG(temp, desc);
			//extractSIFT(temp, desc);
			extractMLBP(temp, desc);
			normalize(desc, desc ,1);
			for(uint pos=0; pos<desc.total(); pos++){
				result.at<float>(point+pos) = desc.at<float>(pos);
			}
			point+=desc.total();
		}
	}
	
	return result;
}

int main(int argc, char** argv)
{
	
	vector<string> trainingPhotos, trainingSketches, testingPhotos, testingSketches, extraPhotos, vphotos, vsketches;
	
	loadImages(argv[1], vphotos);
	loadImages(argv[2], vsketches);
	//loadImages(argv[3], testingPhotos);
	//loadImages(argv[4], testingSketches);
	//loadImages(argv[5], extraPhotos);
	
	auto seed = unsigned (0);
	
	srand (seed);
	random_shuffle (vsketches.begin(), vsketches.end());
	srand (seed);
	random_shuffle (vphotos.begin(), vphotos.end());
	
	trainingPhotos.insert(trainingPhotos.end(),vphotos.begin(),vphotos.begin()+400);
	trainingSketches.insert(trainingSketches.end(),vsketches.begin(),vsketches.begin()+400);
	testingPhotos.insert(testingPhotos.end(),vphotos.begin()+400,vphotos.begin()+600);
	testingSketches.insert(testingSketches.end(),vsketches.begin()+400,vsketches.begin()+600);
	
	//testingPhotos.insert(testingPhotos.end(),extraPhotos.begin(),extraPhotos.begin()+10000);
	
	if(trainingPhotos.size()!=trainingSketches.size()){
		cerr << "Training photos and sketches sets has different sizes" << endl;
		return -1;
	}
	
	uint nTraining = (uint)trainingPhotos.size();
	uint nTestingSketches = testingSketches.size();
	uint nTestingPhotos = testingPhotos.size();
	
	cout << nTraining << " pairs to training." << endl;
	cout << nTestingSketches << " sketches to verify." << endl;
	cout << nTestingPhotos << " photos on the gallery" << endl;
	
	Mat img, temp;
	int size=32, delta=16;
	
	//training
	vector<Mat*> trainingSketchesDescriptors(nTraining), trainingPhotosDescriptors(nTraining);
	
	cout << "extract descriptors from training set" << endl;
	#pragma omp parallel for private(img, temp)
	for(uint i=0; i<nTraining; i++){
		img = imread(trainingSketches[i],0);
		trainingSketchesDescriptors[i] = new Mat();
		
		#pragma omp critical
		temp = extractDescriptors(img, size, delta);
		
		*(trainingSketchesDescriptors[i]) = temp.clone();
	}
	
	#pragma omp parallel for private(img, temp)
	for(uint i=0; i<nTraining; i++){
		img = imread(trainingPhotos[i],0);
		trainingPhotosDescriptors[i] = new Mat();
		
		#pragma omp critical
		temp = extractDescriptors(img, size, delta);
		
		*(trainingPhotosDescriptors[i]) = temp.clone();
	}
	
	//testing
	cout << "extract descriptors from testing set" << endl;
	vector<Mat*> testingSketchesDescriptors(nTestingSketches), testingPhotosDescriptors(nTestingPhotos);
	
	#pragma omp parallel for private(img, temp)
	for(uint i=0; i<nTestingSketches; i++){
		img = imread(testingSketches[i],0);
		testingSketchesDescriptors[i] = new Mat();
		
		#pragma omp critical
		temp = extractDescriptors(img, size, delta);
		
		*(testingSketchesDescriptors[i]) = temp.clone();
	}
	
	#pragma omp parallel for private(img, temp)
	for(uint i=0; i<nTestingPhotos; i++){
		img = imread(testingPhotos[i],0);
		testingPhotosDescriptors[i] = new Mat();
		
		#pragma omp critical
		temp = extractDescriptors(img, size, delta);
		
		*(testingPhotosDescriptors[i]) = temp.clone();
	}
	
	
	PCA pca;
	LDA lda;
	vector<int> labels;
	
	for(uint i=0; i<nTraining; i++){
		labels.push_back(i);
	}
	labels.insert(labels.end(),labels.begin(),labels.end());
	
	//bags
	vector<Mat*> testingSketchesDescriptorsBag(nTestingSketches), testingPhotosDescriptorsBag(nTestingPhotos);
	
	for(int b=0; b<30; b++){
		
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
		
		//crio uma matriz com os descritores que saem da bag
		//aplico uma pca com variancia de .99
		//aplico a lda
		cout << "training pca-lda" << endl;
		
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
			//temp = lda.project(pca.project(temp-meanXs));
			//normalize(temp, temp, 1);
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
			//temp = lda.project(pca.project(temp-meanXp));
			//normalize(temp, temp, 1);
			if(b==0){
				testingPhotosDescriptorsBag[i] = new Mat();
				*(testingPhotosDescriptorsBag[i]) = temp.clone();
			}
			else{
				vconcat(*(testingPhotosDescriptorsBag[i]), temp, *(testingPhotosDescriptorsBag[i]));
			}
		}
		
		cerr << "calculating distances bag: " << b << endl;
		
		
	}
	
	Mat distances = Mat::zeros(nTestingSketches,nTestingPhotos,CV_64F);
	
	#pragma omp parallel for
	for(uint i=0; i<nTestingSketches; i++){
		for(uint j=0; j<nTestingPhotos; j++){
			distances.at<double>(i,j) = abs(1-cosineDistance(*(testingSketchesDescriptorsBag[i]),*(testingPhotosDescriptorsBag[j])));
		}
	}
	
	FileStorage file("kernelproto-drs-cufsf-cosine-dog-mlbp.xml", FileStorage::WRITE);
	file << "distanceMatrix" << distances;
	file.release();
	
	return 0;
}