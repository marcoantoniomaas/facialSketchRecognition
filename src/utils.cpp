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
	vector<vector<Mat> >().swap(result);
	
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
	
	for (int i = 0; i < a.total(); i++){
		double temp = pow((a.at<_Tp>(i) - b.at<_Tp>(i)),2)/(abs(a.at<_Tp>(i)) + abs(b.at<_Tp>(i)));
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

template <typename _Tp> static
inline void cosineDistance_(InputArray _a, InputArray _b, double &dist){
	
	//get matrices
	Mat a = _a.getMat();
	Mat b = _b.getMat();
	
	dist = 0;
	
	for (int i = 0; i < a.total(); i++){
		double temp = a.at<_Tp>(i)*b.at<_Tp>(i);
		dist += temp;
	}
	
	dist = dist/(norm(a)*norm(b));
}

void cosineDistance(InputArray a, InputArray b, double& dist) {
	CV_Assert(a.type() == b.type());
	switch (a.type()) {
		case CV_8SC1:   cosineDistance_<char>(a, b, dist); break;
		case CV_8UC1:   cosineDistance_<unsigned char>(a, b, dist); break;
		case CV_16SC1:  cosineDistance_<short>(a, b, dist); break;
		case CV_16UC1:  cosineDistance_<unsigned short>(a, b, dist); break;
		case CV_32SC1:  cosineDistance_<int>(a, b, dist); break;
		case CV_32FC1:  cosineDistance_<float>(a, b, dist); break;
		case CV_64FC1:  cosineDistance_<double>(a, b, dist); break;
		default: break;
	}
}

double cosineDistance(InputArray a, InputArray b){
	double dist = 0;
	cosineDistance(a, b, dist);
	return dist;
}

vector<int> gen_bag(int tam, double alpha){
	vector<int> temp(tam);
	vector<int> indexes(tam*alpha);
	
	for(int i=0; i<tam; i++){
		temp[i]=i;
	}
	
	srand(time(0));
	random_shuffle(temp.begin(), temp.end());
	
	for(int i=0; i<tam*alpha; i++){
		indexes[i] = temp[i];
	}
	
	return indexes;
}

Mat bag(InputArray desc_, vector<int> &bag_indexes, int tam){
	Mat desc = desc_.getMat();
	int patch_size = desc.rows/tam;
	int bag_size = patch_size*bag_indexes.size();
	Mat result = Mat::zeros(bag_size, 1, desc.type());
	
	for(uint i=0; i<bag_indexes.size(); i++){
		int index = bag_indexes[i];
		Mat temp = desc(Range(index*patch_size,(index+1)*patch_size-1), Range::all());
		temp.copyTo(result(Range(i*patch_size,(i+1)*patch_size-1), Range::all()));
	}
	
	return result;
}

Mat extractDescriptors(InputArray src, int size, int delta, string filter, string descriptor){
	
	Mat img = src.getMat();
	
	if(filter == "DoG")
		img = DoGFilter(img);
	else if(filter == "CSDN")
		img = CSDNFilter(img);
	else if(filter == "Gaussian")
		img = GaussianFilter(img);
	
	int w = img.cols, h=img.rows;
	int n = (w-size)/delta+1, m=(h-size)/delta+1;
	int point = 0;
	
	int descSize = 0;
	
	if(descriptor == "SIFT")
		descSize = 128;
	else if(descriptor == "MLBP")
		descSize = 236;
	else if(descriptor == "HOG")
		descSize = 9;
	else if(descriptor == "HAOG")
		descSize = 9;
	else if(descriptor == "LRBP")
		descSize = 32;
	else if(descriptor == "LBP")
		descSize = 59;
	else
		cout << "No descriptors" << endl;
	
	Mat result = Mat::zeros(m*n*descSize, 1, CV_32F);
	Mat desc, temp;
	
	for(int i=0;i<=w-size;i+=(size-delta)){
		for(int j=0; j<=h-size; j+=(size-delta)){
			temp = img(Rect(i,j,size,size));
			
			if(descriptor == "SIFT")
				extractSIFT(temp,desc);
			else if(descriptor == "MLBP")
				extractMLBP(temp,desc);
			else if(descriptor == "HOG")
				extractHOG(temp,desc);
			else if(descriptor == "HAOG")
				extractHAOG(temp,desc);
			else if(descriptor == "LRBP")
				extractLRBP(temp,desc);
			else if(descriptor == "LBP")
				extractLBP(temp,desc);
			else
				cout << "No descriptors" << endl;
			
			normalize(desc, desc ,1);
			for(uint pos=0; pos<desc.total(); pos++){
				result.at<float>(point+pos) = desc.at<float>(pos);
			}
			point+=desc.total();
		}
	}
	
	return result;
}
