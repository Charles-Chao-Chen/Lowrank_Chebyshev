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

/*
K(x1, y1, x2, y2): kernel between (x1, y1) and (x2, y2)
A[2 * n1]: coordinates of points in a cluster, A[0:n1-1] x coordinates, A[n1:2 *n1-1] y coordinates
U: n1*r array, U n1 x r matrix , column major
V: n2*r array, V n2 x r matrix , column major
S: r x r matrix
*/
void Compute_lowrank
(const Kernel &K, 
int n1, double *A,
int n2, double *B,
int r, double *U, double *S, double *V) {
  Matrix Amat = Eigen::Map<Matrix> (A, n1, 2);
  Matrix Bmat = Eigen::Map<Matrix> (B, n2, 2);

  int r2 = r * r;
  Matrix Umat(n1, r2), Smat(r2, r2), Vmat(n2, r2);
  Compute_lowrank(K, Amat, Bmat, r, Umat, Smat, Vmat);

  // copy data to output
  memcpy(U, Umat.data(), sizeof(double)*n1*r);
  memcpy(S, Smat.data(), sizeof(double)*r2*r2);
  memcpy(V, Vmat.data(), sizeof(double)*n2*r);
}

/*
K(x1, y1, x2, y2): kernel between (x1, y1) and (x2, y2)
A[2 * n1]: coordinates of points in a cluster, A[0:n1-1] x coordinates, A[n1:2 *n1-1] y coordinates
A_lo[2]: lower left corner of box
A_hi[2]: upper right corner of box
U: n1*r array, U n1 x r matrix , column major
V: n2*r array, V n2 x r matrix , column major
S: r x r matrix
*/
void Compute_lowrank
(const Kernel &K, 
int n1, double *A, double *A_lo, double *A_hi,
int n2, double *B, double *B_lo, double *B_hi,
int r, double *U, double *S, double *V) {
  Matrix Amat = Eigen::Map<Matrix> (A, n1, 2);
  Matrix Bmat = Eigen::Map<Matrix> (B, n1, 2);
  //Compute_lowrank();
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

