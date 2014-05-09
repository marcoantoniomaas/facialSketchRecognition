#include <iostream>
#include <set>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	Mat distances;
	
	for(int i=1; i<argc; i++){
		Mat distancesTemp;
		cout << "Reading " << argv[i] << endl; 
		FileStorage file(argv[i], FileStorage::READ);
		if(file.isOpened()==false){
			cerr << "File " << i << " not opened" << endl;
			return -1;
		}
		file["distanceMatrix"] >> distancesTemp;
		normalize(distancesTemp, distancesTemp, 1, 0, NORM_MINMAX);
		if(i==1)
			distances = distancesTemp.clone();
		else
			distances+=distancesTemp;
		file.release();
	}
	
	cout << "The number of subject is: " << distances.rows << " x " << distances.cols << endl; 
	
	vector<int> rank(distances.rows);
	
	for(int i=0; i<distances.rows; i++){
		rank[i] = 1;
		for(int j=0; j<distances.cols; j++){
			if(distances.at<double>(i,j)<=distances.at<double>(i,i) && i!=j){
				rank[i]++;
			}
		}
		//cout << i+1 << ": " << rank[i] << endl;
	}
	
	cout << "rank <- c(";
	for (int i : {1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100}){
		cout << (float)count_if(rank.begin(), rank.end(), [i](int x) {return x <= i;})/distances.rows << ",";
	}
	cout << "\b";
	cout << ")" << endl;
	
	for(int i=0; i<distances.rows; i++) {
		Mat xi = distances.row(i);
		normalize(xi, xi, 1, 0, NORM_MINMAX);
	}
	
	multiset<double> realPairs, impostors;
	
	for(int i=0; i<distances.rows; i++){
		for(int j=0; j<distances.cols; j++){
			if(i==j){
				realPairs.insert(distances.at<double>(i,j));
			}
			else{
				impostors.insert(distances.at<double>(i,j));
			}
		}
	}
	
	cout << "VRatFAR <- c(";
	for (float far : {
		pow(10,-3), pow(10,-2.75), pow(10,-2.5), pow(10,-2.25), 
		 pow(10,-2), pow(10,-1.75), pow(10,-1.5), pow(10,-1.25), 
		 pow(10,-1), pow(10,-0.75), pow(10,-0.5), pow(10,-0.25), 
		 pow(10,0)})
	{
		
		int n = far*impostors.size()-1;
		double threshold;
		multiset<double>::iterator it = impostors.begin();
		if(n>=0){
			advance(it, n);
			threshold = *it;
		}
		else{
			threshold = *it;
		}
		int VR = count_if(realPairs.begin(), realPairs.end(), [threshold](double x) {return x <= threshold;});
		int FR = count_if(impostors.begin(), impostors.end(), [threshold](double x) {return x <= threshold;});
		cout << ((float)VR/distances.rows)*round(((float)n/FR)*10000)/10000.0 << ",";
		
	}
	cout << "\b";
	cout << ")" << endl;
	
	return 0;
}