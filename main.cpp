#include <iostream>
#include <vector>
#include <stdlib.h>   
#include <sys/time.h>
#include <Eigen/Dense>
#include <omp.h>

using std::vector;
using Eigen::MatrixXd;
using Eigen::VectorXd;

typedef unsigned long long timestamp_t;
static timestamp_t get_timestamp () {
  struct timeval now;
  gettimeofday (&now, NULL);
  return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

void print_matrix(vector<vector<double>> matrix) {
  for (vector<double> vect : matrix) {
    for (double x : vect) {
      std::cout <<  x << " ";
    }
    std::cout << "\n";
  }
}

double covxy(const MatrixXd& matrix, int x, int y) {
  double sum = 0.;
  for (int i=0; i<matrix.rows(); i++) {
    sum += (matrix(i,x) * matrix(i,y));
  }
  return sum/(matrix.rows()-1);
}

MatrixXd cov(MatrixXd& matrix) {
  VectorXd means = VectorXd::Zero(matrix.cols());
  
  for (int i=0; i<matrix.rows(); i++) {  
    for (int j=0; j<matrix.cols(); j++) {
	    means(j) += matrix(i,j);
    }
  }
  
  for (int i=0; i<means.size(); i++) {
	  means(i) /= matrix.rows();
  }
  
  #pragma omp parallel for
  for (int j=0; j<matrix.rows(); j++) {  
    for (int i=0; i<matrix.cols(); i++) {
	    matrix(j,i) -= means(i) ;
    }
  }
 
  MatrixXd covmat = MatrixXd::Zero(matrix.cols(),matrix.cols());
  #pragma omp parallel for
  for(int i=0; i<matrix.cols(); i++) {
    for(int j=0; j<i+1; j++) {
      covmat(i,j) = covmat(j,i) = (covxy(matrix,i,j));
    }
  }
 
  return covmat;
}

int main() {
  
  vector<vector<double>> matrix = {{1,1,1}, 
	                           {1,2,1},
                                   {1,3,2},
				   {1,4,3}};
  MatrixXd mtest(4,3);
  mtest << 1,1,1, 
	   1,2,1,
	   1,3,2,
	   1,4,3;


  
  MatrixXd mat = MatrixXd::Random(900000,12);
  MatrixXd mat2 = MatrixXd::Random(900000,12);

  std::cout << "started" << std::endl;
  timestamp_t t0 = get_timestamp();
  
  MatrixXd covmat = cov(mat);
  
  timestamp_t t1 = get_timestamp();
  double secs = (t1 - t0) / 1000000.0L;
  std::cout << "ended" << std::endl;
  std::cout << secs << std::endl;
  

  timestamp_t t2 = get_timestamp();
  MatrixXd centered = mat2.rowwise() - mat2.colwise().mean();
  MatrixXd cov1 = (centered.adjoint() * centered) / double(mat2.rows() - 1); 
  timestamp_t t3 = get_timestamp();
  double secs2 = (t3 - t2) / 1000000.0L;
  std::cout << secs2 << std::endl;
  MatrixXd mt = cov(mtest);
  std::cout << mt << std::endl;
  
  return 0;
}











