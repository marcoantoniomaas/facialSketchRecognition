#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
#include "descriptors.hpp"
#include "utils.hpp"
#include "filters.hpp"
#include "kernelproto.hpp"

using namespace std;
using namespace cv;

Mat extractDescriptors(InputArray src, int size, int delta){
	
	Mat img = src.getMat();
	//img = DoGFilter(img);
	int w = img.cols, h=img.rows;
	int n = (w-size)/delta+1, m=(h-size)/delta+1;
	int point = 0;
	
	Mat result = Mat::zeros(m*n*128, 1, CV_32F);
	Mat desc, temp;
	
	for(int i=0;i<=w-size;i+=(size-delta)){
		for(int j=0; j<=h-size; j+=(size-delta)){
			temp = img(Rect(i,j,size,size));
			//extractHAOG(temp, desc);
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
	
	vector<string> trainingPhotos1, trainingSketches1, trainingPhotos2, trainingSketches2, 
	testingPhotos, testingSketches, extraPhotos, vphotos, vsketches;
	
	loadImages(argv[1], vphotos);
	loadImages(argv[2], vsketches);
	//loadImages(argv[3], testingPhotos);
	//loadImages(argv[4], testingSketches);
	//loadImages(argv[5], extraPhotos);
	
	if(vphotos.size()!=vsketches.size()){
		cerr << "Training photos and sketches sets has different sizes" << endl;
		return -1;
	}
	
	auto seed = unsigned (0);
	
	srand (seed);
	random_shuffle (vsketches.begin(), vsketches.end());
	srand (seed);
	random_shuffle (vphotos.begin(), vphotos.end());
	
	int tam = 159;
	
	trainingPhotos1.insert(trainingPhotos1.end(),vphotos.begin(),vphotos.begin()+tam/3);
	trainingSketches1.insert(trainingSketches1.end(),vsketches.begin(),vsketches.begin()+tam/3);
	trainingPhotos2.insert(trainingPhotos2.end(),vphotos.begin()+tam/3,vphotos.begin()+2*tam/3);
	trainingSketches2.insert(trainingSketches2.end(),vsketches.begin()+tam/3,vsketches.begin()+2*tam/3);
	testingPhotos.insert(testingPhotos.end(),vphotos.begin()+2*tam/3,vphotos.begin()+tam);
	testingSketches.insert(testingSketches.end(),vsketches.begin()+2*tam/3,vsketches.begin()+tam);
	
	//testingPhotos.insert(testingPhotos.end(),extraPhotos.begin(),extraPhotos.begin()+10000);
	
	uint nTraining1 = (uint)trainingPhotos1.size();
	uint nTraining2 = (uint)trainingPhotos2.size();
	uint nTestingSketches = testingSketches.size();
	uint nTestingPhotos = testingPhotos.size();
	
	cout << nTraining1 << " and " << nTraining2 <<  " pairs to training." << endl;
	cout << nTestingSketches << " sketches to verify." << endl;
	cout << nTestingPhotos << " photos on the gallery" << endl;
	
	Mat img, temp;
	int size=32, delta=16;
	
	//training
	vector<Mat*> trainingSketchesDescriptors1(nTraining1), trainingPhotosDescriptors1(nTraining1), 
	trainingSketchesDescriptors2(nTraining2), trainingPhotosDescriptors2(nTraining2);
	
	cout << "extract descriptors from training set" << endl;
	#pragma omp parallel for private(img, temp)
	for(uint i=0; i<nTraining1; i++){
		img = imread(trainingSketches1[i],0);
		trainingSketchesDescriptors1[i] = new Mat();
		
		#pragma omp critical
		temp = extractDescriptors(img, size, delta);
		
		*(trainingSketchesDescriptors1[i]) = temp.clone();
	}
	
	#pragma omp parallel for private(img, temp)
	for(uint i=0; i<nTraining1; i++){
		img = imread(trainingPhotos1[i],0);
		trainingPhotosDescriptors1[i] = new Mat();
		
		#pragma omp critical
		temp = extractDescriptors(img, size, delta);
		
		*(trainingPhotosDescriptors1[i]) = temp.clone();
	}

	#pragma omp parallel for private(img, temp)
	for(uint i=0; i<nTraining2; i++){
		img = imread(trainingSketches2[i],0);
		trainingSketchesDescriptors2[i] = new Mat();
		
		#pragma omp critical
		temp = extractDescriptors(img, size, delta);
		
		*(trainingSketchesDescriptors2[i]) = temp.clone();
	}
	
	#pragma omp parallel for private(img, temp)
	for(uint i=0; i<nTraining2; i++){
		img = imread(trainingPhotos2[i],0);
		trainingPhotosDescriptors2[i] = new Mat();
		
		#pragma omp critical
		temp = extractDescriptors(img, size, delta);
		
		*(trainingPhotosDescriptors2[i]) = temp.clone();
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
	
	for(uint i=0; i<nTraining2; i++){
		labels.push_back(i);
	}
	labels.insert(labels.end(),labels.begin(),labels.end());
	
	//bags
	vector<Mat*> testingSketchesDescriptorsBag(nTestingSketches), testingPhotosDescriptorsBag(nTestingPhotos), 
	trainingPhotosDescriptors1Temp(nTraining1), trainingSketchesDescriptors1Temp(nTraining1);
	
	for(int b=0; b<30; b++){
		
		vector<int> bag_indexes = gen_bag(154, 0.1);
		
		#pragma omp parallel for private(img, temp)
		for(uint i=0; i<nTraining1; i++){
			temp = *(trainingSketchesDescriptors1[i]);
			temp = bag(temp, bag_indexes, 154);
			trainingSketchesDescriptors1Temp[i] = new Mat();
			*(trainingSketchesDescriptors1Temp[i]) = temp.clone();
		}
		
		#pragma omp parallel for private(img, temp)
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
		
		#pragma omp parallel for private(img, temp)
		for(uint i=0; i<nTraining2; i++){
			temp = *(trainingSketchesDescriptors2[i]);
			temp = bag(temp, bag_indexes, 154);
			temp = k.projectProbe(temp);
			temp.copyTo(X.col(i));
		}
		
		#pragma omp parallel for private(img, temp)
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
		
		
		//crio uma matriz com os descritores que saem da bag
		//aplico uma pca com variancia de .99
		//aplico a lda
		cout << "training pca-lda" << endl;
		
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
			temp = k.projectProbe(temp);
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
			temp = k.projectGallery(temp);
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
	
	FileStorage file("kernelproto-prs-cufsf-cosine3.xml", FileStorage::WRITE);
	file << "distanceMatrix" << distances;
	file.release();
	
	return 0;
}