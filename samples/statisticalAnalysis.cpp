#include <iostream>
#include <set>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	Mat distances;
	FileStorage file(argv[1], FileStorage::READ);
	
	if(file.isOpened()==false){
		cerr << "No file was opened" << endl;
		return -1;
	}
	file["distanceMatrix"] >> distances;
	file.release();
	
	vector<int> rank(distances.rows);
	multiset<double> realPairs, impostors;
	
	for(int i=0; i<distances.rows; i++){
		rank[i] = 1;
		for(int j=0; j<distances.cols; j++){
			if(i==j){
				realPairs.insert(distances.at<double>(i,j));
			}
			else{
				impostors.insert(distances.at<double>(i,j));
				if(distances.at<double>(i,j)<=distances.at<double>(i,i)){
					rank[i]++;
				}
			}
		}
		cout << i+1 << ": " << rank[i] << endl;
	}
	
	cout << "The number of subject is: " << distances.rows << endl; 
	
	for (int i : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100}){
		cout << "Rank "<< i << ": ";
		cout << "d= " << (float)count_if(rank.begin(), rank.end(), [i](int x) {return x <= i;})/distances.rows*100 << "%" << endl;
	}
	
	for (float far : {0.0001, 0.001, 0.01, 0.1, 1.0}){
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
		cout << "VR at FAR " << far*100 << "%:\t" << (float)count_if(realPairs.begin(), realPairs.end(), [threshold](double x) 
		{return x <= threshold;})/distances.rows*100 << "%" << endl;
	}

	
	return 0;
}