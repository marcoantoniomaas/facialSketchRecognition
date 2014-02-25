#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Eigenvalues>
#include <algorithm>
//#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace Eigen;
using namespace cv;
using namespace boost::filesystem;

void loadImages(string src, vector<string>& dest);
void patcher(InputArray src, Size size, int delta, vector<vector<Mat> >& patches);
void chiSquareDistance(InputArray a, InputArray b, double& dist);
double chiSquareDistance(InputArray a, InputArray b);
void cosineDistance(InputArray a, InputArray b, double& dist);
double cosineDistance(InputArray a, InputArray b);
vector<int> gen_bag(int tam, double alpha);
Mat bag(InputArray desc, vector<int> &bag_indexes, int tam);

#endif
