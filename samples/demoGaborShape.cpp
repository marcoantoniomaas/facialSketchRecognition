#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
#include "filters.hpp"
#include "transforms.hpp"
#include "descriptors.hpp"
#include "utils.hpp"

using namespace std;
using namespace cv;

Mat extractGaborShape(InputArray _img){
	/// Set the number of patches
	int mhor = 5, mver = 7, nhor = 6, nver = 2;
	/// Establish the number of bins
	int histSize = 8;
	/// Set the ranges of histogram
	float range[] = {0, 1} ;
	const float* histRange = { range };
	
	Mat img = _img.getMat();
	vector<vector<Mat> > mpatches, rpatches;
	int count = 0;
	Mat hist, radon, gaborMag;
	Mat temp = Mat::zeros(40*mhor*mver*nhor*nver*histSize, 1, CV_32F);
	
	for(int mu=0; mu<8; mu++){
		for(int nu=0; nu<5; nu++){
			gaborMag = magnitude(convolveDFT(img, gaborWavelet(mu, nu, 2*CV_PI, 0)));
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

#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
#include "descriptors.hpp"
#include "utils.hpp"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	string descriptor = "GS";
	string database = "CUFSF";
	
	uint nTraining = 500;
	uint pcaDim = 700;
	uint count = 1;
	
	vector<string> trainingPhotos, trainingSketches, testingPhotos, testingSketches, extraPhotos, photos, sketches;
	
	loadImages(argv[1], photos);
	loadImages(argv[2], sketches);
	
	auto seed = unsigned(count);
	
	srand (seed);
	random_shuffle (sketches.begin(), sketches.end());
	srand (seed);
	random_shuffle (photos.begin(), photos.end());
	
	trainingPhotos.insert(trainingPhotos.end(), photos.begin(),photos.begin()+nTraining);
	trainingSketches.insert(trainingSketches.end(), sketches.begin(), sketches.begin()+nTraining);
	testingPhotos.insert(testingPhotos.end(),photos.begin()+nTraining,photos.end());
	testingSketches.insert(testingSketches.end(),sketches.begin()+nTraining,sketches.end());
	
	//testingPhotos.insert(testingPhotos.end(),extraPhotos.begin(),extraPhotos.begin()+10000);
	
	uint nTestingSketches = testingSketches.size();
	uint nTestingPhotos = testingPhotos.size();
	
	cout << nTestingSketches << " sketches to verify." << endl;
	cout << nTestingPhotos << " photos on the gallery" << endl;
	
	Mat img, temp;
	
	//training 
	vector<int> labels;
	
	for(uint i=0; i<2*nTraining; i++)
		labels.push_back(i%nTraining);
	
	int dim = 40*5*7*6*2*8;
	
	Mat data = Mat::zeros(2*nTraining, dim, CV_32F);
	
	PCA pca;
	LDA lda;
	
	#pragma omp parallel for private(img, temp)
	for(uint i=0; i<nTraining; i++){
		img = imread(trainingSketches[i],0);
		
		temp = extractGaborShape(img);
		
		temp = temp.t();
		
		temp.copyTo(data.row(i));
		
		cout << "trainingSketches " << i << endl;
	}
	
	#pragma omp parallel for private(img, temp)
	for(uint i=0; i<nTraining; i++){
		img = imread(trainingPhotos[i],0);
		
		temp = extractGaborShape(img);
		
		temp = temp.t();
		
		temp.copyTo(data.row(i+nTraining));
		
		cout << "trainingPhotos " << i << endl;
	}
	
	if(nTraining>0){
		pca(data, Mat(), CV_PCA_DATA_AS_ROW, pcaDim);
		lda.compute(pca.project(data), labels);
	}
	
	//testing
	vector<Mat*> testingSketchesDescriptors(nTestingSketches), testingPhotosDescriptors(nTestingPhotos);
	
	#pragma omp parallel for private(img, temp)
	for(uint i=0; i<nTestingSketches; i++){
		img = imread(testingSketches[i],0);
		
		testingSketchesDescriptors[i] = new Mat();
		
		temp = extractGaborShape(img);
		
		if(nTraining>0)
			*(testingSketchesDescriptors[i]) = lda.project(pca.project(temp.t()));
		else
			*(testingSketchesDescriptors[i]) = temp.clone();
		
		cout << "testingSketches " << i << endl;
	}
	
	#pragma omp parallel for private(img, temp)
	for(uint i=0; i<nTestingPhotos; i++){
		img = imread(testingPhotos[i],0);
		
		testingPhotosDescriptors[i] = new Mat();
		
		temp = extractGaborShape(img);
		
		if(nTraining>0)
			*(testingPhotosDescriptors[i]) = lda.project(pca.project(temp.t()));
		else
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
	
	
	string file1name = descriptor + database + to_string(nTraining) + "-" + to_string(pcaDim) + string("chi") + to_string(count) + string(".xml");
	string file2name = descriptor + database + to_string(nTraining) + "-" + to_string(pcaDim) + string("l2") + to_string(count) + string(".xml");
	string file3name = descriptor + database + to_string(nTraining) + "-" + to_string(pcaDim) + string("cosine") + to_string(count) + string(".xml");
	
	FileStorage file1(file1name, FileStorage::WRITE);
	FileStorage file2(file2name, FileStorage::WRITE);
	FileStorage file3(file3name, FileStorage::WRITE);
	
	file1 << "distanceMatrix" << distancesChi;
	file2 << "distanceMatrix" << distancesL2;
	file3 << "distanceMatrix" << distancesCosine;
	
	file1.release();
	file2.release();
	file3.release();
	
	return 0;
}