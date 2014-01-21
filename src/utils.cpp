#include "utils.hpp"

void loadImages(string src, vector<string> &dest){
	directory_iterator end;
	
	path dir(src);
	
	string filename;
	int n=0;
	int num;
	
	for (directory_iterator pos(dir); pos != end; ++pos){
		if(is_regular_file(*pos)){
			n++;
		}
	}
	
	dest.resize(n);
	
	for (directory_iterator pos(dir); pos != end; ++pos){
		if(is_regular_file(*pos)){
			filename = pos->path().filename().string();
			num = atoi((filename.substr(0,filename.find("."))).c_str());
			dest[num-1] = string(pos->path().c_str());
		}
	}
}

void patcher(InputArray src, Size size, int delta, vector<vector<Mat> > &result){
	
	Mat img = src.getMat();
	int w = img.cols, h=img.rows, ww=size.width, hh=size.height;
	
	for(int i=0;i<=w-ww;i+=(ww-delta)){
		vector<Mat> col;
		for(int j=0;j<=h-hh;j+=(hh-delta)){
			col.push_back(img(Rect(i,j,ww,hh)));
		}
		result.push_back(col);
	}
	
}

template <typename _Tp> static
inline void chiSquareDistance_(InputArray _a, InputArray _b, double &dist){
	
	//get matrices
	Mat a = _a.getMat();
	Mat b = _b.getMat();
	
	for (int i = 0; i < a.rows; i++){
		double temp = pow((a.at<_Tp>(i) - b.at<_Tp>(i)),2)/(a.at<_Tp>(i) + b.at<_Tp>(i));
		if(temp==temp)
			dist += temp;
	}
}

void chiSquareDistance(InputArray a, InputArray b, double& dist) {
	CV_Assert(a.type() == b.type());
	switch (a.type()) {
		case CV_8SC1:   chiSquareDistance_<char>(a, b, dist); break;
		case CV_8UC1:   chiSquareDistance_<unsigned char>(a, b, dist); break;
		case CV_16SC1:  chiSquareDistance_<short>(a, b, dist); break;
		case CV_16UC1:  chiSquareDistance_<unsigned short>(a, b, dist); break;
		case CV_32SC1:  chiSquareDistance_<int>(a, b, dist); break;
		case CV_32FC1:  chiSquareDistance_<float>(a, b, dist); break;
		case CV_64FC1:  chiSquareDistance_<double>(a, b, dist); break;
		default: break;
	}
}

double chiSquareDistance(InputArray a, InputArray b){
	double dist = 0;
	chiSquareDistance(a, b, dist);
	return dist;
}
