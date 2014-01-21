#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include "filters.hpp"
#include "transforms.hpp"
#include "descriptors.hpp"
#include "utils.hpp"

using namespace std;
using namespace cv;

int main( int argc, char** argv )
{
	
	vector<string> trainingPhotos, trainingSketches, testingPhotos, testingSketches, extraPhotos;
	
	loadImages(argv[1], trainingPhotos);
	loadImages(argv[2], trainingSketches);
	loadImages(argv[3], testingPhotos);
	loadImages(argv[4], testingSketches);
	loadImages(argv[5], extraPhotos);
	
	const int nTraining = (const int) trainingPhotos.size();
	vector<Mat> trainingPhotosGaborShape(nTraining), trainingSketchesGaborShape(nTraining), testingPhotosGaborShape, testingSketchesGaborShape, extraPhotosGaborShape;
	
	cout << trainingPhotos.size() << endl;
	cout << trainingSketches.size() << endl;
	
	/// Set the number of patches
	int mhor = 5, mver = 7, nhor = 6, nver = 2;
	/// Establish the number of bins
	int histSize = 8;
	/// Set the ranges of histogram
	float range[] = {0, 1} ;
	const float* histRange = { range };
	
	#pragma omp parallel for
	for(uint i=0; i<nTraining; i++){
		Mat img;
		vector<vector<Mat>> mpatches, rpatches;
		cout << "trainingPhoto " << i << endl; 
		img = imread(trainingPhotos[i],0);
		Mat temp;
		for(int mu=0; mu<8; mu++)
			for(int nu=0; nu<5; nu++){
				Mat gaborMag = magnitude(convolveDFT(img, gaborWavelet(mu, nu, 2*CV_PI, 21)));
				patcher(gaborMag, Size(gaborMag.cols/mhor, gaborMag.rows/mver), 0, mpatches);
				for(uint mcol=0; mcol<mpatches.size(); mcol++)
					for(uint mrow=0; mrow<mpatches[0].size(); mrow++){
						Mat radon = radonTransform(mpatches[mcol][mrow]);
						patcher(radon, Size(radon.cols/nhor, radon.rows/nver), 0, rpatches);
						for(uint rcol=0; rcol<rpatches.size(); rcol++)
							for(uint rrow=0; rrow<rpatches[0].size(); rrow++){
								Mat hist;
								calcHist(&rpatches[rcol][rrow], 1, 0, Mat(), hist, 1, &histSize, &histRange);
								if(temp.empty())
									temp = hist.clone();
								else
									vconcat(temp, hist, temp);
							}
							rpatches.clear();	
					}
					mpatches.clear();	
			}
			trainingPhotosGaborShape[i]=temp.clone();
			temp.release();
	}
	
	#pragma omp parallel for
	for(uint i=0; i<nTraining; i++){
		Mat img;
		vector<vector<Mat>> mpatches, rpatches;
		cout << "trainingSketches " << i << endl; 
		img = imread(trainingSketches[i],0);
		Mat temp;
		for(int mu=0; mu<8; mu++)
			for(int nu=0; nu<5; nu++){
				Mat gaborMag = magnitude(convolveDFT(img, gaborWavelet(mu, nu, 2*CV_PI, 21)));
				patcher(gaborMag, Size(gaborMag.cols/mhor, gaborMag.rows/mver), 0, mpatches);
				for(uint mcol=0; mcol<mpatches.size(); mcol++)
					for(uint mrow=0; mrow<mpatches[0].size(); mrow++){
						Mat radon = radonTransform(mpatches[mcol][mrow]);
						patcher(radon, Size(radon.cols/nhor, radon.rows/nver), 0, rpatches);
						for(uint rcol=0; rcol<rpatches.size(); rcol++)
							for(uint rrow=0; rrow<rpatches[0].size(); rrow++){
								Mat hist;
								calcHist(&rpatches[rcol][rrow], 1, 0, Mat(), hist, 1, &histSize, &histRange);
								if(temp.empty())
									temp = hist.clone();
								else
									vconcat(temp, hist, temp);
							}
							rpatches.clear();	
					}
					mpatches.clear();	
			}
			trainingSketchesGaborShape[i]=temp.clone();
			temp.release();
	}
	
	
	int nTestingSketches = trainingSketchesGaborShape.size();
	int nTestingPhotos = trainingPhotosGaborShape.size();
	
	vector<int> rank(nTestingSketches);
	
	cerr << "calculating distances" << endl;
	
	for(int i=0; i<nTestingSketches; i++){
		double val = chiSquareDistance(trainingPhotosGaborShape[i], trainingSketchesGaborShape[i]);
		cerr << "photo and sketch "<< i << " d1= "<< val << endl;
		int temp = 0;
		for(int j=0; j<nTestingPhotos; j++){
			if(chiSquareDistance(trainingPhotosGaborShape[j],trainingSketchesGaborShape[i])<= val && i!=j){
				cerr << "small "<< j << " d1= "<< chiSquareDistance(trainingPhotosGaborShape[j],trainingSketchesGaborShape[i]) << endl;
				temp++;
			}
		}
		rank[i] = temp;
		cerr << i << " rank= " << temp << endl;
	}
	
	for (int i : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100})
	{
		cerr << "Rank "<< i << ": ";
		cerr << "d1= " << (float)count_if(rank.begin(), rank.end(), [i](int x) {return x < i;})/nTestingSketches << endl;
	}
	
	return 0;
}

