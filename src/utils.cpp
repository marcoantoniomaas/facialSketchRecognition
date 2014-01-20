#include "utils.hpp"

void loadImages(string src, vector<string> &dest){
  directory_iterator end;
  
  path dir(src);
  
  string filename;
  int n=0;
  int num;
  Mat temp;
  
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

void patcher(Mat img, int size, int delta, vector<vector<Mat> > &result){
  int w = img.cols, h=img.rows;
  
  for(int i=0;i<=h-size;i+=(size-delta)){
    vector<Mat> col;
    for(int j=0;j<=w-size;j+=(size-delta)){
      col.push_back(img(Rect(j,i,size,size)));
    }
    result.push_back(col);
  }
  
}

float chiSquareDistance(Mat a, Mat b){
  float result = 0;
  for (int i = 0; i < a.rows; i++){
    float temp = pow((a.at<float>(i) - b.at<float>(i)),2)/(a.at<float>(i) + b.at<float>(i));
    if(temp==temp)
      result += temp;
  }
  return result;
}