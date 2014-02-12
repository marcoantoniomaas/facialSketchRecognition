#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include "filters.hpp"
#include "transforms.hpp"
#include "descriptors.hpp"
#include "utils.hpp"

using namespace std;
using namespace cv;

/// Establish the number of bins
int histSize = 32;
/// Set the ranges of histogram
float range[] = {0, 255} ;
const float* histRange = { range };

Mat extractLRBP(InputArray _img, int level){
	
	Mat img = _img.getMat();
	vector<vector<Mat> > patches;
	int count = 0;
	Mat hist, radon, lrbp;
	Mat temp = Mat::zeros(1, pow(2, 2*level)*histSize, CV_32F);
	
	patcher(img, Size(img.cols/pow(2,level), img.rows/pow(2,level)), 0, patches);
	
	//cout << patches.size() << endl;
	//cout << patches[0].size() << endl;
	
	for(uint mcol=0; mcol<patches.size(); mcol++){
		for(uint mrow=0; mrow<patches[mcol].size(); mrow++){
			radon = radonTransform(patches[mcol][mrow]);
			lrbp = elbp(radon, 2, 8);
			lrbp.convertTo(lrbp, CV_32F);
			calcHist(&lrbp, 1, 0, Mat(), hist, 1, &histSize, &histRange);
			for(int pos=0; pos<histSize; pos++){
				temp.at<float>(count*histSize+pos) = hist.at<float>(pos);
			}
			count++;
		}
	}
	//cout << count*histSize << endl;
	//cout << temp.size() << endl;
	
	patches.clear();
	vector<vector<Mat> >().swap(patches);
	
	return temp;
}


int main(int argc, char** argv)
{
	
	vector<string> trainingPhotos, trainingSketches, testingPhotos, testingSketches, extraPhotos, vphotos, vsketches;
	
	loadImages(argv[1], vphotos);
	loadImages(argv[2], vsketches);
	//loadImages(argv[3], testingPhotos);
	//loadImages(argv[4], testingSketches);
	//loadImages(argv[5], extraPhotos);
	
	testingPhotos.insert(testingPhotos.end(),vphotos.begin(),vphotos.end());
	testingSketches.insert(testingSketches.end(),vsketches.begin(),vsketches.end());
	
	vector<Mat*> testingPhotosGaborShape, testingSketchesGaborShape, extraPhotosGaborShape;
	Mat img, temp;
	
	uint nTestingSketches = testingSketches.size();
	uint nTestingPhotos = testingPhotos.size();
	
	testingSketchesGaborShape.resize(nTestingSketches); 
	testingPhotosGaborShape.resize(nTestingPhotos); 
	
	cout << "The number of subjects is: " << nTestingSketches << endl;
	cout << "The number of photos in gallery is: "<< nTestingPhotos << endl;
	
	Mat distances = Mat::zeros(nTestingSketches,nTestingPhotos,CV_64F);
	FileStorage file("lrbp-cuhk.xml", FileStorage::WRITE);
	
	for(uint level=0; level<5; level++){
		
		#pragma omp parallel for private(img, temp)
		for(uint i=0; i<nTestingPhotos; i++){
			img = imread(testingPhotos[i],0);
			//resize(img, img, Size(128,160));
			cout << "testingPhotos " << i << endl;
			testingPhotosGaborShape[i] = new Mat();
			temp = extractLRBP(img,level);
			//normalize(temp, temp, 1, 0, NORM_MINMAX);
			*(testingPhotosGaborShape[i]) = temp;
			//cout << i<<" "<<*(testingPhotosGaborShape[i]) << endl;
		}
		
		#pragma omp parallel for private(img, temp)
		for(uint i=0; i<nTestingSketches; i++){
			img = imread(testingSketches[i],0);
			//resize(img, img, Size(128,160));
			cout << "testingSketches " << i << endl;
			testingSketchesGaborShape[i] = new Mat();
			temp = extractLRBP(img,level);
			//normalize(temp, temp, 1, 0, NORM_MINMAX);
			*(testingSketchesGaborShape[i]) = temp;
			//cout << i <<" "<<*(testingSketchesGaborShape[i]) << endl;
		}
		
		//for(uint i=0; i<nTestingSketches; i++){
		//	cout <<*(testingPhotosGaborShape[i]) << endl;
		//	cout <<*(testingSketchesGaborShape[i]) << endl;
		//}
		
		cerr << "calculating distances" << endl;
		
		#pragma omp parallel for
		for(uint i=0; i<nTestingSketches; i++){
			for(uint j=0; j<nTestingPhotos; j++){
				distances.at<double>(i,j) += chiSquareDistance(*(testingPhotosGaborShape[j]),*(testingSketchesGaborShape[i]))/pow(2,5-level+1);//chiSquareDistance();
			}
		}
	}
	
	file << "distanceMatrix" << distances;
	file.release();
	
	return 0;
}