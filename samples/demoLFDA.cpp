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
	
	vector<vector<Mat> > patches;
	patcher(img, Size(size,size), delta, patches);
	
	Mat result = Mat::zeros(m*n*364, 1, CV_32F);
	Mat a, b, temp;
	
	for(uint i=0; i<patches.size(); i++){
		for(uint j=0; j<patches[0].size(); j++){
			temp = patches[i][j];
			extractSIFT(temp,a);
			normalize(a,a,1);
			for(uint pos=0; pos<a.total(); pos++){
				result.at<float>(point+pos) = a.at<float>(pos);
			}
			point+=a.total();
			
			extractMLBP(temp,b);
			normalize(b,b,1);
			
			for(uint pos=0; pos<b.total(); pos++){
				result.at<float>(point+pos) = b.at<float>(pos);
			}
			point+=b.total();
		}
	}
	
	return result;
}

int main(int argc, char** argv)
{
	string filter = "Gaussian";
	string descriptor = "HOG";
	string database = "CUFS-extra";
	uint descSize = 9;
	uint count = 1;
	
	vector<string> extraPhotos, photos, sketches;
	
	loadImages(argv[3], photos);
	loadImages(argv[4], sketches);
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
	vector<Mat*> trainingSketchesDescriptors, trainingPhotosDescriptors;
	
	trainingSketchesDescriptors.insert(trainingSketchesDescriptors.end(), sketchesDescriptors.begin(), sketchesDescriptors.begin()+nTraining);
	trainingPhotosDescriptors.insert(trainingPhotosDescriptors.end(), photosDescriptors.begin(), photosDescriptors.begin()+nTraining);
	
	//testing
	vector<Mat*> testingSketchesDescriptors, testingPhotosDescriptors;
	
	testingSketchesDescriptors.insert(testingSketchesDescriptors.end(), sketchesDescriptors.begin()+nTraining, sketchesDescriptors.end());
	testingPhotosDescriptors.insert(testingPhotosDescriptors.end(), photosDescriptors.begin()+nTraining, photosDescriptors.end());
	testingPhotosDescriptors.insert(testingPhotosDescriptors.end(), extraDescriptors.begin(), extraDescriptors.end());
	
	uint nTestingSketches = testingSketchesDescriptors.size(),
	nTestingPhotos = testingPhotosDescriptors.size();
	
	img = imread(photos[0],0);
	uint n = (img.cols-size)/delta+1, m=(img.rows-size)/delta+1;
	uint dim = (*(trainingPhotosDescriptors[0])).total();
	
	vector<Mat*> projectionMatrix(n);
	Mat Xs(dim, nTraining, CV_32F), Xp(dim, nTraining, CV_32F), X(dim, 2*nTraining, CV_32F);
	PCA pca;
	
	#pragma omp parallel for private(temp)
	for(uint i=0; i<nTraining; i++){
		temp = *(trainingSketchesDescriptors[i]);
		temp.copyTo(Xs.col(i));
		temp.copyTo(X.col(i));
	}
	
	#pragma omp parallel for private(temp)
	for(uint i=0; i<nTraining; i++){
		temp = *(trainingPhotosDescriptors[i]);
		temp.copyTo(Xp.col(i));
		temp.copyTo(X.col(i+nTraining));
	}
	
	Mat meanXs = Mat::zeros(dim, 1, CV_32F),
	meanXp = Mat::zeros(dim, 1, CV_32F),
	meanX = Mat::zeros(dim, 1, CV_32F), instance;
	
	// calculate sums
	for (int i = 0; i < Xs.cols; i++) {
		instance = Xs.col(i);
		add(meanXs, instance, meanXs);
	}
	for (int i = 0; i < Xp.cols; i++) {
		instance = Xp.col(i);
		add(meanXp, instance, meanXp);
	}
	for (int i = 0; i < X.cols; i++) {
		instance = X.col(i);
		add(meanX, instance, meanX);
	}
	
	// calculate total mean
	meanXs.convertTo(meanXs, CV_32F, 1.0/static_cast<double>(Xs.cols));
	meanXp.convertTo(meanXp, CV_32F, 1.0/static_cast<double>(Xp.cols));
	meanX.convertTo(meanX, CV_32F, 1.0/static_cast<double>(X.cols));
	
	// subtract the mean of matrix
	for(int i=0; i<Xs.cols; i++) {
		Mat c_i = Xs.col(i);
		subtract(c_i, meanXs.reshape(1,dim), c_i);
	}
	
	for(int i=0; i<Xp.cols; i++) {
		Mat c_i = Xp.col(i);
		subtract(c_i, meanXp.reshape(1,dim), c_i);
	}
	
	for(int i=0; i<X.cols; i++) {
		Mat c_i = X.col(i);
		subtract(c_i, meanX.reshape(1,dim), c_i);
	}
	
	for(uint i=0; i<n; i++){
		Range slice = Range(i*m*descSize,(i+1)*m*descSize);
		
		pca(X(slice, Range::all()), Mat(), CV_PCA_DATA_AS_COL, 100);
		
		Mat W = (pca.eigenvectors.clone()).t();
		
		Mat Y = W.t()*(Xs(slice , Range::all())+ Xp(slice, Range::all()))/2;
		
		Mat XXs = W.t()*Xs(slice, Range::all())-Y;
		Mat XXp = W.t()*Xp(slice, Range::all())-Y;
		Mat XX;
		
		hconcat(XXs, XXp, XX);
		
		pca(XX, Mat(), CV_PCA_DATA_AS_COL, 100);
		
		Mat diag = Mat::diag(pca.eigenvalues);
		sqrt(diag.inv(DECOMP_SVD),diag);
		
		Mat V = (diag*pca.eigenvectors).t();
		
		pca(V.t()*Y, Mat(), CV_PCA_DATA_AS_COL, 99);
		
		Mat U = (pca.eigenvectors.clone()).t();
		
		projectionMatrix[i] = new Mat();
		*(projectionMatrix[i]) = W*V*U;
	}
	
	//testing
	
	vector<Mat*> testingSketchesProjection(nTestingSketches), testingPhotosProjection(nTestingPhotos);
	
	#pragma omp parallel for private(temp)
	for(uint i=0; i<nTestingSketches; i++){
		Mat desc(1, n*99, CV_32F);
		
		temp = *(testingSketchesDescriptors[i]);
		temp = temp - meanXs;
		
		for(uint col=0; col<n; col++){
			Range slice = Range(col*m*descSize,(col+1)*m*descSize);
			Mat aux = ((*(projectionMatrix[col])).t()*temp(slice, Range::all())).t();
			aux.copyTo(desc(Range::all(), Range(col*99,(col+1)*99)));
		}
		testingSketchesProjection[i] = new Mat();
		*(testingSketchesProjection[i]) = desc.clone();
	}
	
	#pragma omp parallel for private(temp)
	for(uint i=0; i<nTestingPhotos; i++){
		Mat desc(1, n*99, CV_32F);
		
		temp = *(testingPhotosDescriptors[i]);
		temp = temp - meanXp;
		
		for(uint col=0; col<n; col++){
			Range slice = Range(col*m*descSize,(col+1)*m*descSize);
			Mat aux = ((*(projectionMatrix[col])).t()*temp(slice, Range::all())).t();
			aux.copyTo(desc(Range::all(), Range(col*99,(col+1)*99)));
		}
		testingPhotosProjection[i] = new Mat();
		*(testingPhotosProjection[i]) = desc.clone();
	}
	
	Mat distancesL2 = Mat::zeros(nTestingSketches,nTestingPhotos,CV_64F);
	Mat distancesCosine = Mat::zeros(nTestingSketches,nTestingPhotos,CV_64F);
	
	#pragma omp parallel for
	for(uint i=0; i<nTestingSketches; i++){
		for(uint j=0; j<nTestingPhotos; j++){
			distancesL2.at<double>(i,j) = norm(*(testingSketchesProjection[i]),*(testingPhotosProjection[j]));
			distancesCosine.at<double>(i,j) = abs(1-cosineDistance(*(testingSketchesProjection[i]),*(testingPhotosProjection[j])));
		}
	}
	
	string file1name = "LFDA-" + to_string(size) + filter + descriptor + database + to_string(nTraining) + string("l2") + to_string(count) + string(".xml");
	string file2name = "LFDA-" + to_string(size) + filter + descriptor + database + to_string(nTraining) + string("cosine") + to_string(count) + string(".xml");
	
	FileStorage file1(file1name, FileStorage::WRITE);
	FileStorage file2(file2name, FileStorage::WRITE);
	
	file1 << "distanceMatrix" << distancesL2;
	file2 << "distanceMatrix" << distancesCosine;
	
	file1.release();
	file2.release();
	
	return 0;
}