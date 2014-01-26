#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include "filters.hpp"
#include "transforms.hpp"
#include "descriptors.hpp"
#include "utils.hpp"

using namespace std;
using namespace cv;

/// Set the number of patches
int mhor = 5, mver = 7, nhor = 6, nver = 2;
/// Establish the number of bins
int histSize = 8;
/// Set the ranges of histogram
float range[] = {0, 1} ;
const float* histRange = { range };

Mat extractGaborShape(InputArray _img){
	
	Mat img = _img.getMat();
	vector<vector<Mat> > mpatches, rpatches;
	int count = 0;
	Mat hist, radon, gaborMag;
	Mat temp = Mat::zeros(40*mhor*mver*nhor*nver*histSize, 1, CV_32F);
	
	for(int mu=0; mu<8; mu++){
		for(int nu=0; nu<5; nu++){
			gaborMag = magnitude(convolveDFT(img, gaborWavelet(mu, nu, 2*CV_PI, 21)));
			patcher(gaborMag, Size(gaborMag.cols/mhor, gaborMag.rows/mver), 0, mpatches);
			for(uint mcol=0; mcol<mpatches.size(); mcol++){
				for(uint mrow=0; mrow<mpatches[mcol].size(); mrow++){
					radon = radonTransform(mpatches[mcol][mrow]);
					patcher(radon, Size(radon.cols/nhor, radon.rows/nver), 0, rpatches);
					for(uint rcol=0; rcol<rpatches.size(); rcol++){
						for(uint rrow=0; rrow<rpatches[rcol].size(); rrow++){
							calcHist(&rpatches[rcol][rrow], 1, 0, Mat(), hist, 1, &histSize, &histRange);
							for(int pos=0; pos<histSize; pos++){
								temp.at<float>(count*histSize+pos) = hist.at<float>(pos);
							}
							count++;
						}
					}
					rpatches.clear();
					vector<vector<Mat> >().swap(rpatches);
				}
			}
			mpatches.clear();
			vector<vector<Mat> >().swap(mpatches);
		}
	}
	return temp;
}


int main( int argc, char** argv )
{
	
	vector<string> trainingPhotos, trainingSketches, testingPhotos, testingSketches, extraPhotos;
	
	loadImages(argv[3], trainingPhotos);
	loadImages(argv[4], trainingSketches);
	loadImages(argv[3], testingPhotos);
	loadImages(argv[4], testingSketches);
	loadImages(argv[5], extraPhotos);
	
	uint nTraining = (uint)trainingPhotos.size();
	vector<Mat*> trainingPhotosGaborShape, trainingSketchesGaborShape, testingPhotosGaborShape, testingSketchesGaborShape, extraPhotosGaborShape;
	trainingPhotosGaborShape.resize(nTraining);
	trainingSketchesGaborShape.resize(nTraining);
	
	cout << trainingPhotos.size() << endl;
	cout << trainingSketches.size() << endl;
	
	Mat img;
	
	#pragma omp parallel for private(img)
	for(uint i=0; i<nTraining; i++){
		img = imread(trainingPhotos[i],0);;
		cout << "trainingPhoto " << i << endl;
		trainingPhotosGaborShape[i] = new Mat();
		*(trainingPhotosGaborShape[i]) = extractGaborShape(img);
	}
	
	#pragma omp parallel for private(img)
	for(uint i=0; i<nTraining; i++){
		img = imread(trainingSketches[i],0);;
		cout << "trainingSketches " << i << endl;
		trainingSketchesGaborShape[i] = new Mat();
		*(trainingSketchesGaborShape[i]) = extractGaborShape(img);
	}
	
	int nTestingSketches = trainingSketchesGaborShape.size();
	int nTestingPhotos = trainingPhotosGaborShape.size();
	
	cerr << "calculating distances" << endl;
	
	Mat distances = Mat::zeros(nTestingSketches,nTestingPhotos,CV_64F);
	FileStorage file("distances.xml", FileStorage::WRITE);
	
	#pragma omp parallel for
	for(int i=0; i<nTestingSketches; i++){
		for(int j=0; j<nTestingPhotos; j++){
			distances.at<double>(i,j) = chiSquareDistance(*(trainingPhotosGaborShape[j]),*(trainingSketchesGaborShape[i]));
		}
	}
	
	file << "distanceMatrix" << distances;
	file.release();
	
	return 0;
}

