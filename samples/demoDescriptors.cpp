#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
#include "descriptors.hpp"
#include "utils.hpp"

using namespace std;
using namespace cv;

Mat extractDescriptors(InputArray src, int size, int delta){
	
	Mat img = src.getMat();
	int w = img.cols, h=img.rows;
	int n = delta==0? w/size:(w-size)/delta+1, m= delta==0? h/size:(h-size)/delta+1;
	int point = 0;
	
	vector<vector<Mat> > patches;
	patcher(img, Size(size,size), delta, patches);
	
	Mat result = Mat::zeros(m*n*32, 1, CV_32F);
	Mat a, b, temp;
	
	for(uint i=0; i<patches.size(); i++){
		for(uint j=0; j<patches[0].size(); j++){
			temp = patches[i][j];
			//extractSIFT(temp,a);
			//extractMLBP(temp,a);
			extractLRBP(temp, a);
			//extractHAOG(temp, a);
			normalize(a,a,1);
			for(uint pos=0; pos<a.total(); pos++){
				result.at<float>(point+pos) = a.at<float>(pos);
			}
			point+=a.total();
		}
	}
	
	return result;
}

int main(int argc, char** argv)
{
	
	vector<string> testingPhotos, testingSketches, extraPhotos, photos, sketches;
	
	loadImages(argv[1], photos);
	loadImages(argv[2], sketches);
	
	testingPhotos.insert(testingPhotos.end(),photos.begin(),photos.end());
	testingSketches.insert(testingSketches.end(),sketches.begin(),sketches.end());
	
	//testingPhotos.insert(testingPhotos.end(),extraPhotos.begin(),extraPhotos.begin()+10000);
	
	uint nTestingSketches = testingSketches.size();
	uint nTestingPhotos = testingPhotos.size();
	
	cout << nTestingSketches << " sketches to verify." << endl;
	cout << nTestingPhotos << " photos on the gallery" << endl;
	
	Mat img, temp;
	int size = 32;
	int delta = size/2;
	
	//testing
	vector<Mat*> testingSketchesDescriptors(nTestingSketches), testingPhotosDescriptors(nTestingPhotos);
	
	#pragma omp parallel for private(img, temp)
	for(uint i=0; i<nTestingSketches; i++){
		img = imread(testingSketches[i],0);
		//resize(img, img, Size(64,80));
		testingSketchesDescriptors[i] = new Mat();
		
		#pragma omp critical
		temp = extractDescriptors(img, size, delta);
		
		*(testingSketchesDescriptors[i]) =temp.clone();
		
		cout << "testingSketches " << i << endl;
	}
	
	#pragma omp parallel for private(img, temp)
	for(uint i=0; i<nTestingPhotos; i++){
		img = imread(testingPhotos[i],0);
		//resize(img, img, Size(64,80));
		testingPhotosDescriptors[i] = new Mat();
		
		#pragma omp critical
		temp = extractDescriptors(img, size, delta);
		
		*(testingPhotosDescriptors[i]) = temp.clone();
		
		cout << "testingPhotos " << i << endl;
		//cout << *(testingPhotosDescriptors[i]) << endl;
	}
	
	cerr << "calculating distances" << endl;
	
	
	Mat distancesChi = Mat::zeros(nTestingSketches,nTestingPhotos,CV_64F);
	Mat distancesL2 = Mat::zeros(nTestingSketches,nTestingPhotos,CV_64F);
	Mat distancesCosine = Mat::zeros(nTestingSketches,nTestingPhotos,CV_64F);
	
	#pragma omp parallel for
	for(uint i=0; i<nTestingSketches; i++){
		for(uint j=0; j<nTestingPhotos; j++){
			distancesChi.at<double>(i,j) = chiSquareDistance(*(testingSketchesDescriptors[i]),*(testingPhotosDescriptors[j]));
			distancesL2.at<double>(i,j) = norm(*(testingSketchesDescriptors[i]),*(testingPhotosDescriptors[j]));
			distancesCosine.at<double>(i,j) = abs(1-cosineDistance(*(testingSketchesDescriptors[i]),*(testingPhotosDescriptors[j])));
		}
	}
	
	
	FileStorage file1("lrbp-cufsf-chi.xml", FileStorage::WRITE);
	FileStorage file2("lrbp-cufsf-l2.xml", FileStorage::WRITE);
	FileStorage file3("lrbp-cufsf-cosine.xml", FileStorage::WRITE);
	
	file1 << "distanceMatrix" << distancesChi;
	file2 << "distanceMatrix" << distancesL2;
	file3 << "distanceMatrix" << distancesCosine;
	
	file1.release();
	file2.release();
	file3.release();
	
	return 0;
}