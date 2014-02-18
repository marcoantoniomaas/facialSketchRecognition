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
	int w = img.cols, h=img.rows;
	int n = (w-size)/delta+1, m=(h-size)/delta+1;
	int point = 0;
	
	Mat result = Mat::zeros(m*n*128, 1, CV_32F);
	Mat desc, temp;
	
	for(int i=0;i<=w-size;i+=(size-delta)){
		for(int j=0; j<=h-size; j+=(size-delta)){
			temp = img(Rect(i,j,size,size));
			extractSIFT(temp, desc);
			//extractMLBP(temp, desc);
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
	
	auto seed = unsigned (time(0));
	
	srand (seed);
	random_shuffle (vsketches.begin(), vsketches.end());
	srand (seed);
	random_shuffle (vphotos.begin(), vphotos.end());
	
	trainingPhotos.insert(trainingPhotos.end(),vphotos.begin()+994,vphotos.begin()+1194);
	trainingSketches.insert(trainingSketches.end(),vsketches.begin()+994,vsketches.begin()+1194);
	testingPhotos.insert(testingPhotos.end(),vphotos.begin(),vphotos.begin()+100);
	testingSketches.insert(testingSketches.end(),vsketches.begin(),vsketches.begin()+100);
	
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
		img = DoGFilter(img);
		trainingSketchesDescriptors[i] = new Mat();
		
		#pragma omp critical
		temp = extractDescriptors(img, size, delta);
		
		*(trainingSketchesDescriptors[i]) = temp.clone();
	}
	
	#pragma omp parallel for private(img, temp)
	for(uint i=0; i<nTraining; i++){
		img = imread(trainingPhotos[i],0);
		img = DoGFilter(img);
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
		img = DoGFilter(img);
		testingSketchesDescriptors[i] = new Mat();
		
		#pragma omp critical
		temp = extractDescriptors(img, size, delta);
		
		*(testingSketchesDescriptors[i]) = temp.clone();
	}
	
	#pragma omp parallel for private(img, temp)
	for(uint i=0; i<nTestingPhotos; i++){
		img = imread(testingPhotos[i],0);
		img = DoGFilter(img);
		testingPhotosDescriptors[i] = new Mat();
		
		#pragma omp critical
		temp = extractDescriptors(img, size, delta);
		
		*(testingPhotosDescriptors[i]) = temp.clone();
	}
	
	
	PCA pca;
	LDA lda;
	Mat distances = Mat::zeros(nTestingSketches,nTestingPhotos,CV_64F);
	vector<int> labels;
	
	for(uint i=0; i<nTraining; i++){
		labels.push_back(i);
	}
	labels.insert(labels.end(),labels.begin(),labels.end());
	
	//bags
	for(int b=0; b<1; b++){
		//crio uma bag
		
		uint dim = (*(trainingPhotosDescriptors[0])).total();
		
		Mat X(dim, 2*nTraining, CV_32F);
		
		
		#pragma omp parallel for private(img, temp)
		for(uint i=0; i<nTraining; i++){
			temp = *(trainingSketchesDescriptors[i]);
			//extract bag
			temp.copyTo(X.col(i));
		}
		
		#pragma omp parallel for private(img, temp)
		for(uint i=0; i<nTraining; i++){
			temp = *(trainingPhotosDescriptors[i]);
			//extract bag
			temp.copyTo(X.col(i+nTraining));
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
		
		//crio uma matriz com os descritores que saem da bag
		//aplico uma pca com variancia de .99
		//aplico a lda
		cout << "training pca-lda" << endl;
		pca.computeVar(X, Mat(), CV_PCA_DATA_AS_COL, .99);
		Mat W1 = (pca.eigenvectors.clone()).t();
		Mat ldaData = (W1.t()*X).t();
		lda.compute(ldaData, labels);
		Mat W2 = lda.eigenvectors();
		W2.convertTo(W2, CV_32F);
		Mat projectionMatrix = (W2.t()*W1.t()).t();
		
		
		//testing
		
		vector<Mat*> testingSketchesDescriptorsBag(nTestingSketches), testingPhotosDescriptorsBag(nTestingPhotos);
		
		#pragma omp parallel for private(img, temp)
		for(uint i=0; i<nTestingSketches; i++){
			testingSketchesDescriptorsBag[i] = new Mat();
			temp = *(testingSketchesDescriptors[i]);//extract bag and multiply by projectionMatrix
			*(testingSketchesDescriptorsBag[i]) = projectionMatrix.t()*(temp-meanX);
		}
		
		#pragma omp parallel for private(img, temp)
		for(uint i=0; i<nTestingPhotos; i++){
			testingPhotosDescriptorsBag[i] = new Mat();
			temp = *(testingPhotosDescriptors[i]);//extract bag and multiply by projectionMatrix
			*(testingPhotosDescriptorsBag[i]) = projectionMatrix.t()*(temp-meanX);
		}
		
		cerr << "calculating distances bag: " << b << endl;
		
		#pragma omp parallel for
		for(uint i=0; i<nTestingSketches; i++){
			for(uint j=0; j<nTestingPhotos; j++){
				distances.at<double>(i,j) += norm(*(testingSketchesDescriptorsBag[i]),*(testingPhotosDescriptorsBag[j]));
			}
		}
	}
	
	FileStorage file("kernelproto-drs-cufsf.xml", FileStorage::WRITE);
	file << "distanceMatrix" << distances;
	file.release();
	
	return 0;
}