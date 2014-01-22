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

Mat extractGaborShape(Mat img){
	vector<vector<Mat> > mpatches, rpatches;
	Mat temp;
	for(int mu=0; mu<8; mu++){
		for(int nu=0; nu<5; nu++){
			Mat gaborMag = magnitude(convolveDFT(img, gaborWavelet(mu, nu, 2*CV_PI, 21)));
			patcher(gaborMag, Size(gaborMag.cols/mhor, gaborMag.rows/mver), 0, mpatches);
			for(uint mcol=0; mcol<mpatches.size(); mcol++){
				for(uint mrow=0; mrow<mpatches[0].size(); mrow++){
					Mat radon = radonTransform(mpatches[mcol][mrow]);
					patcher(radon, Size(radon.cols/nhor, radon.rows/nver), 0, rpatches);
					for(uint rcol=0; rcol<rpatches.size(); rcol++){
						for(uint rrow=0; rrow<rpatches[0].size(); rrow++){
							Mat hist;
							calcHist(&rpatches[rcol][rrow], 1, 0, Mat(), hist, 1, &histSize, &histRange);
							if(temp.empty())
								temp = hist.clone();
							else
								vconcat(temp, hist, temp);
						}
					}
					vector<vector<Mat> >().swap(rpatches);
				}
			}
			vector<vector<Mat> >().swap(mpatches);
		}
	}
	return temp.clone();
}


int main( int argc, char** argv )
{
	
	vector<string> trainingPhotos, trainingSketches, testingPhotos, testingSketches, extraPhotos;
	
	loadImages(argv[1], trainingPhotos);
	loadImages(argv[2], trainingSketches);
	loadImages(argv[3], testingPhotos);
	loadImages(argv[4], testingSketches);
	loadImages(argv[5], extraPhotos);
	
	const uint nTraining = (const uint) trainingPhotos.size();
	vector<Mat> trainingPhotosGaborShape(nTraining), trainingSketchesGaborShape(nTraining), testingPhotosGaborShape, testingSketchesGaborShape, extraPhotosGaborShape;
	
	cout << trainingPhotos.size() << endl;
	cout << trainingSketches.size() << endl;
	
	Mat img;
	
	#pragma omp parallel for private(img)
	for(uint i=0; i<nTraining; i++){
		img = imread(trainingPhotos[i],0);;
		cout << "trainingPhoto " << i << endl; 
		trainingPhotosGaborShape[i] =  extractGaborShape(img);
	}
	
	#pragma omp parallel for private(img)
	for(uint i=0; i<nTraining; i++){
		img = imread(trainingSketches[i],0);;
		cout << "trainingSketches " << i << endl; 
		trainingSketchesGaborShape[i] =  extractGaborShape(img);
	}
	
	int nTestingSketches = trainingSketchesGaborShape.size();
	int nTestingPhotos = trainingPhotosGaborShape.size();
	
	vector<int> rank(nTestingSketches);
	
	cerr << "calculating distances" << endl;
	
	Mat distances = Mat::zeros(nTestingSketches,nTestingPhotos,CV_64F);
	FileStorage file("distances.xml", FileStorage::WRITE);
	
	for(int i=0; i<nTestingSketches; i++){
		for(int j=0; j<nTestingPhotos; j++){
			distances.at<double>(i,j) = chiSquareDistance(trainingPhotosGaborShape[j],trainingSketchesGaborShape[i]);
		}
	}
	
	file << "distanceMatrix" << distances;
	file.release();
	
	return 0;
}

