#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Eigenvalues>
//#include <opencv2/core/eigen.hpp>

using namespace Eigen;
using namespace cv;
using namespace boost::filesystem;

void loadImages(string, vector<string>&);
float chiSquareDistance(Mat, Mat);
void patcher(Mat, int, int, vector<vector<Mat> >&);

#endif
