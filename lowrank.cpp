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
 
Vector Chebyshev_points(int n, double a, double b) {
  return Chebyshev_points(n)*(b-a)/2.0 + Vector::Constant(n, (a+b)/2.0);
}

class Interpolation {
public:
  Interpolation(double a_, double b_, int r) {
    a = a_;
    b = b_;
    Cheb = Chebyshev_points(r)*(b-a)/2.0 + Vector::Constant(r, (a+b)/2.0);
  }
  Interpolation() {}

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

void Compute_lowrank_1d
(FUN1D K, 
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

class Interpolation_2d {
public:
  Interpolation_2d(double lo_x, double lo_y, 
                   double hi_x, double hi_y, int r_) {
    this->r = r_;
    Ix = Interpolation(lo_x, hi_x, r);
    Iy = Interpolation(lo_y, hi_y, r);
  }

  // get 2d interpolation nodes
  Matrix GetChebyshev() const {
    Vector Cheb_x = Ix.GetChebyshev();
    Vector Cheb_y = Iy.GetChebyshev();
    Matrix Cheb_2d(r*r, 2);
    for (int i=0; i<r; i++) 
      for (int j=0; j<r; j++)
        Cheb_2d.row(i*r+j) << Cheb_x(i), Cheb_y(j);
    return Cheb_2d;  
  }

  // compute interpoloation matrix
  Matrix ComputeMatrix(const Matrix &X) const {
    int N = X.rows();
    Matrix U(N, r*r);
    Matrix Ux = Ix.ComputeMatrix(X.col(0));  
    Matrix Uy = Iy.ComputeMatrix(X.col(1));  
    for (int i=0; i<r; i++)
      for (int j=0; j<r; j++) 
        for (int k=0; k<N; k++)
          U(k, i*r+j) = Ux(k, i) * Uy(k, j);
    return U;
  }

private:
  int r;
  Interpolation Ix;
  Interpolation Iy;
};

void Compute_lowrank
(const Kernel &K, const Matrix &A, const Matrix &B,
const int r, Matrix &U, Matrix &S, Matrix &V) {

  Interpolation_2d IA(A.col(0).minCoeff(), A.col(1).minCoeff(),
                      A.col(0).maxCoeff(), A.col(1).maxCoeff(), r);
  Interpolation_2d IB(B.col(0).minCoeff(), B.col(1).minCoeff(),
                      B.col(0).maxCoeff(), B.col(1).maxCoeff(), r);

  U = IA.ComputeMatrix(A);
  V = IB.ComputeMatrix(B);

  Matrix chebA = IA.GetChebyshev();
  Matrix chebB = IB.GetChebyshev();

  int r2 = r*r;
  for (int i=0; i<r2; i++)
    for (int j=0; j<r2; j++)
      S(j, i) = K( chebA.row(j), chebB.row(i) );
}

void Compute_lowrank
(FUN2D K, const Matrix &A, const Matrix &B,
const int r, Matrix &U, Matrix &S, Matrix &V) {

  Interpolation_2d IA(A.col(0).minCoeff(), A.col(1).minCoeff(),
                      A.col(0).maxCoeff(), A.col(1).maxCoeff(), r);
  Interpolation_2d IB(B.col(0).minCoeff(), B.col(1).minCoeff(),
                      B.col(0).maxCoeff(), B.col(1).maxCoeff(), r);

  U = IA.ComputeMatrix(A);
  V = IB.ComputeMatrix(B);

  Matrix chebA = IA.GetChebyshev();
  Matrix chebB = IB.GetChebyshev();

  int r2 = r*r;
  for (int i=0; i<r2; i++)
    for (int j=0; j<r2; j++)
      S(j, i) = K( chebA(j,1), chebA(j,2), chebB(i,1), chebB(i,2) );
}

/* Interface to Julia */
typedef double (*FUN1D)(double, double);
typedef double (*FUN2D)(double, double,double, double);

extern "C" void bbfmm1D(FUN1D kfun, double*X, double*Y, double xmin, double xmax, double ymin, double ymax, 
      double *U, double*V, int r, int n1, int n2){
        Vector Vx(n1), Vy(n2);
        for(int i=0;i<n1;i++) Vx(i) = X[i];
        for(int i=0;i<n2;i++) Vy(i) = Y[i];
        Matrix mU(n1,r), mS(r,r), mV(n2,r);
        Compute_lowrank_1d(kfun, Vx, xmin, xmax, Vy, ymin, ymax, r, mU, mS, mV);
        mU = mU*mS;
        for(int i=0;i<mU.rows()*mU.cols();i++){
          U[i] = mU(i);
        }
        for(int i=0;i<mV.rows()*mV.cols();i++){
          V[i] = mV(i);
        }
}

extern "C" void bbfmm2D(FUN2D kfun, double*X, double*Y, double xmin, double xmax, double ymin, double ymax, 
      double *U, double*V, int r, int n1, int n2){
        Matrix Vx(n1,2), Vy(n2,2);
        for(int i=0;i<n1;i++) Vx(i,1) = X[i];
        for(int i=0;i<n2;i++) Vy(i,1) = Y[i];
        for(int i=0;i<n1;i++) Vx(i,2) = X[i+n1];
        for(int i=0;i<n2;i++) Vy(i,2) = Y[i+n2];
        Matrix mU(n1,r), mS(r,r), mV(n2,r);
        Compute_lowrank(kfun, Vx, Vy, r, mU, mS, mV);
        mU = mU*mS;
        for(int i=0;i<mU.rows()*mU.cols();i++){
          U[i] = mU(i);
        }
        for(int i=0;i<mV.rows()*mV.cols();i++){
          V[i] = mV(i);
        }
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
  Compute_lowrank(K, X, Y, r, U, S, V);

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

