#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <Eigen/Core>

typedef Eigen::MatrixXd Matrix;
typedef Eigen::VectorXd Vector;

class Kernel {
public:
  Kernel() {};
  double operator() (const Vector &x, const Vector &y) const {
    //return (x-y).squaredNorm();
    return 1.0/((x-y).squaredNorm());
  }
  double operator() (double x, double y) const {
    return 1.0/((x-y)*(x-y));
  }
};

// Chebyshev points in [-1, 1]
Vector Chebyshev_points(int n) {
  Vector X(n);
  for (int i=0; i<n; i++)
    X(i) = cos( M_PI*(2.0*i+1.0)/(2.0*n) );
  return X;
}
  
class Interpolation {
public:
  Interpolation(double a_, double b_, int r) {
    a = a_;
    b = b_;
    Cheb = Chebyshev_points(r)*(b-a)/2.0 + Vector::Constant(r, (a+b)/2.0);
  }
  Vector GetChebyshev() const {return Cheb;}
  Matrix ComputeMatrix(const Vector &X) const {
    int N = X.size();
    int r = Cheb.size();
    Matrix P(N, r);
    for (int i=0; i<N; i++)
      for (int j=0; j<r; j++) {
        P(i, j) = 1.0;
        for (int k=0; k<r; k++) {
	 if (k != j)
	   P(i, j) *= (X(i)-Cheb(k))/(Cheb(j)-Cheb(k));
	}
      }
    return P;
  }
private:
  double a;
  double b;
  Vector Cheb;
};

void Compute_lowrank_1d
(const Kernel &K, 
const Vector &X, double xmin, double xmax, 
const Vector &Y, double ymin, double ymax,
const int r, Matrix &U, Matrix &S, Matrix &V) {
  
  Interpolation Ix(xmin, xmax, r);
  Interpolation Iy(ymin, ymax, r);
  U = Ix.ComputeMatrix(X);
  V = Iy.ComputeMatrix(Y);
  Vector chebX = Ix.GetChebyshev();
  Vector chebY = Iy.GetChebyshev();
  for (int i=0; i<r; i++)
    for (int j=0; j<r; j++)
      S(i, j) = K( chebX(i), chebY(j) );
}

void Compute_lowrank
(const Kernel &K,
const Matrix &X, double xmin, double xmax,
const Matrix &Y, double ymin, double ymax,
const int r, Matrix &U, Matrix &S, Matrix &V) {

  int n = X.rows();
  Matrix Ux(n, r), Uy(n, r), Vx(n, r), Vy(n, r);
  Interpolation Ix(xmin, xmax, r);
  Interpolation Iy(ymin, ymax, r);
  Ux = Ix.ComputeMatrix(X.col(0));
  Uy = Ix.ComputeMatrix(X.col(1));
  Vx = Iy.ComputeMatrix(Y.col(0));
  Vy = Iy.ComputeMatrix(Y.col(1));
 
  for (int i=0; i<r; i++)
    for (int j=0; j<r; j++) 
      for (int k=0; k<n; k++) {
        U(k, i*r+j) = Ux(k, i) * Uy(k, j);
        V(k, i*r+j) = Vx(k, i) * Vy(k, j);
      }

  int r2 = r*r;
  Matrix chebX(r2, 2);
  Matrix chebY(r2, 2);
  Vector chebX_1d = Ix.GetChebyshev();
  Vector chebY_1d = Iy.GetChebyshev();
  for (int i=0; i<r; i++) 
    for (int j=0; j<r; j++) {
      chebX.row(i*r+j) << chebX_1d(i), chebX_1d(j);
      chebY.row(i*r+j) << chebY_1d(i), chebY_1d(j);
    }

  for (int i=0; i<r2; i++)
    for (int j=0; j<r2; j++)
      S(i, j) = K( chebX.row(i), chebY.row(j) );
}

int main(int argc, char *argv[]) {

  int n = 5;
  int r = 2;

  for (int i=1; i<argc; i++) {
    if (!strcmp(argv[i], "-n"))
      n = atoi(argv[++i]);
    if (!strcmp(argv[i], "-r"))
      r = atoi(argv[++i]);
  }
  std::cout<<"Number of points: "<<n<<std::endl
           <<"Chebyshev points (each dimension): "<<r
	   <<std::endl;

  // random 2D points
  Matrix X = Matrix::Random(n, 2); // [-1, 1]^2
  Matrix Y = Matrix::Random(n, 2) + Matrix::Constant(n, 2, 10); // [9, 11]^2

  Kernel K;
  Matrix A(n, n);
  for (int i=0; i<n; i++) {
    for (int j=0; j<n; j++) {
      A(i, j) = K( X.row(i), Y.row(j) );
    }
  }

  int r2 = r*r;
  Matrix U(n, r2), S(r2, r2), V(n, r2);
  Compute_lowrank(K, X, -1, 1, Y, 9, 11, r, U, S, V);

  Matrix E = A - U*S*V.transpose();

#if 0
  std::cout<<"Hello world!"<<std::endl;
  std::cout<<"X: "<<X<<std::endl;
  std::cout<<"Y: "<<Y<<std::endl;
  std::cout<<"Matrix: \n"<<A<<std::endl;
#endif

  std::cout<<"Error of lowrank: \n"<<E.norm()<<std::endl;
  return 0;
}

