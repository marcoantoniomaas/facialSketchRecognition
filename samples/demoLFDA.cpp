#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include "descriptors.hpp"
#include "utils.hpp"

using namespace std;
using namespace cv;

Mat extractDescriptors(InputArray src, int size, int delta){
	
	Mat img = src.getMat();
	int w = img.cols, h=img.rows;
	int n = (w-size)/delta+1, m=(h-size)/delta+1;
	int point = 0;
	
	Mat result = Mat::zeros(m*n*364, 1, CV_32F);
	Mat a, b, temp;
	
	for(int i=0;i<=w-size;i+=(size-delta)){
		for(int j=0; j<=h-size; j+=(size-delta)){
			temp = img(Rect(i,j,size,size));
			calcSIFTDescriptors(temp,a);
			normalize(a,a,1);
			for(uint pos=0; pos<a.total(); pos++){
				result.at<float>(point+pos) = a.at<float>(pos);
			}
			point+=a.total();
			
			calcLBPHistogram(temp,b);
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
	
	vector<string> trainingPhotos, trainingSketches, testingPhotos, testingSketches, extraPhotos, vphotos, vsketches;
	
	loadImages(argv[1], vphotos);
	loadImages(argv[2], vsketches);
	//loadImages(argv[3], testingPhotos);
	//loadImages(argv[4], testingSketches);
	//loadImages(argv[5], extraPhotos);
	
	trainingPhotos.insert(trainingPhotos.end(),vphotos.begin()+694,vphotos.begin()+1194);
	trainingSketches.insert(trainingSketches.end(),vsketches.begin()+694,vsketches.begin()+1194);
	testingPhotos.insert(testingPhotos.end(),vphotos.begin(),vphotos.begin()+694);
	testingSketches.insert(testingSketches.end(),vsketches.begin(),vsketches.begin()+694);
	
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
	
	
	//training
	Mat img, temp;
	int size=16, delta=8;
	
	img = imread(trainingSketches[0],0);
	//resize(img, img, Size(64,80));
	temp = extractDescriptors(img, size, delta);
	
	int n = (img.cols-size)/delta+1, m=(img.rows-size)/delta+1;
	uint dim = temp.total();
	
	vector<Mat*> projectionMatrix(n);
	Mat Xs(dim, nTraining, CV_32F), Xp(dim, nTraining, CV_32F), X(dim, 2*nTraining, CV_32F);
	
	PCA pca;
	
	#pragma omp parallel for private(img, temp)
	for(uint i=0; i<nTraining; i++){
		img = imread(trainingSketches[i],0);
		//resize(img, img, Size(64,80));
		#pragma omp critical
		temp = extractDescriptors(img, size, delta);
		temp.copyTo(Xs.col(i));
		temp.copyTo(X.col(i));
		cout << "trainingSketches " << i << endl;
	}
	
	#pragma omp parallel for private(img, temp)
	for(uint i=0; i<nTraining; i++){
		img = imread(trainingPhotos[i],0);
		//resize(img, img, Size(64,80));
		#pragma omp critical
		temp = extractDescriptors(img, size, delta);
		temp.copyTo(Xp.col(i));
		temp.copyTo(X.col(i+nTraining));
		cout << "trainingPhoto " << i << endl;
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
		subtract(c_i, meanXp.reshape(1,dim), c_i);
	}
	
	for(int i=0; i<Xp.cols; i++) {
		Mat c_i = Xp.col(i);
		subtract(c_i, meanXp.reshape(1,dim), c_i);
	}
	
	for(int i=0; i<X.cols; i++) {
		Mat c_i = X.col(i);
		subtract(c_i, meanXp.reshape(1,dim), c_i);
	}
	
	for(int i=0; i<n; i++){
		Range slice = Range(i*m*364, (i+1)*m*364);
		
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
	
	vector<Mat*> testingSketchesDescriptors(nTestingSketches), testingPhotosDescriptors(nTestingPhotos);
	
	for(uint i=0; i<nTestingSketches; i++){
		Mat desc(1, n*99, CV_32F);
		img = imread(testingSketches[i],0);
		//resize(img, img, Size(64,80));
		testingSketchesDescriptors[i] = new Mat();
		temp = extractDescriptors(img, size, delta);
		
		for(int i=0; i<n; i++){
			Mat aux = ((*(projectionMatrix[i])).t()*temp(Range(i*m*364,(i+1)*m*364), Range::all())).t();
			//normalize(aux,aux,1,0, NORM_MINMAX);
			aux.copyTo(desc(Range::all(), Range(i*99,(i+1)*99)));
		}
		
		*(testingSketchesDescriptors[i]) = desc.clone();
		
		cout << "testingSketches " << i << endl;
		//cout << *(testingSketchesDescriptors[i]) << endl;
	}
	
	for(uint i=0; i<nTestingPhotos; i++){
		Mat desc(1, n*99, CV_32F);
		img = imread(testingPhotos[i],0);
		//resize(img, img, Size(64,80));
		testingPhotosDescriptors[i] = new Mat();
		temp = extractDescriptors(img, size, delta);
		
		for(int i=0; i<n; i++){
			Mat aux = ((*(projectionMatrix[i])).t()*temp(Range(i*m*364,(i+1)*m*364), Range::all())).t();
			//normalize(aux,aux,1,0, NORM_MINMAX);
			aux.copyTo(desc(Range::all(), Range(i*99,(i+1)*99)));
		}
		
		*(testingPhotosDescriptors[i]) = desc.clone();
		
		cout << "testingPhotos " << i << endl;
		//cout << *(testingPhotosDescriptors[i]) << endl;
	}
	
	cerr << "calculating distances" << endl;
	
	Mat distances = Mat::zeros(nTestingSketches,nTestingPhotos,CV_64F);
	FileStorage file("lfda-500cufsf694-final16.xml", FileStorage::WRITE);
	
	#pragma omp parallel for
	for(uint i=0; i<nTestingSketches; i++){
		for(uint j=0; j<nTestingPhotos; j++){
			distances.at<double>(i,j) = norm(*(testingSketchesDescriptors[i]),*(testingPhotosDescriptors[j]));//chiSquareDistance();
		}
	}
	
	file << "distanceMatrix" << distances;
	file.release();
	
	return 0;
}