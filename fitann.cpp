#include <cstdio>
#include <fstream>

#include <Eigen/Dense>
#include <iostream>
#include <cstdio>

#include "nn.cpp"

typedef Eigen::Matrix<f_type, Eigen::Dynamic, Eigen::Dynamic> matrix_t;
typedef Eigen::Matrix<f_type, Eigen::Dynamic, 1> vector_t;
typedef Eigen::Array<f_type, Eigen::Dynamic, Eigen::Dynamic> array_t;

int main (int argc, const char* argv[]) 
{
   // input dimensionality
   int n_input = 2;
   // output dimensionality
   int n_output = 1;
   // number of training samples
   int m = 961;
   // number of evaluation samples
   int k = 961;
       
   // training inputs
   matrix_t X(m, n_input);
   matrix_t Y(m, n_output);
   matrix_t Z(m, n_input);

   // throw away stuff
   double eater;
   
   std::ifstream file("adp_ann_example", std::ios::in);
	//std::ifstream file("adp_abf_example", std::ios::in);

   for(int i = 0; i < m; ++i)
   {
       file >> X(i, 0);
	   file >> X(i, 1);

	   //file >> eater;
	
       file >> Y(i,0);

	   file >> Z(i,0);	
	   file >> Z(i,1);	
   }

   file.close();
   
   Eigen::VectorXi topo(4);
   topo << n_input, 12, 6, n_output;

   nnet::neural_net nn(topo);
   nn.set_train_params({0.005, 1.e10, 10.0, 1.e-7, 0, 50});

   nn.autoscale(X,Y);

   nn.train(X,Y, true);
   //nn.train(X,Y,Z, true);

   nn.forward_pass(X);

   matrix_t YY = nn.get_activation();
   std::ofstream file4("output.txt");
   file4 << YY << std::endl;
   file4.close();

   matrix_t dX = nn.get_gradient_forwardpass(0);
   std::ofstream file5("dx.txt");
   file5 << dX << std::endl;
   file5.close();

   matrix_t dY = nn.get_gradient_forwardpass(1);
   std::ofstream file6("dy.txt");
   file6 << dY << std::endl;
   file6.close();

   std::ofstream file7("dxy_bp.txt");
   for(int i = 0; i < m; ++i)
   		file7 << nn.get_gradient(i) << std::endl;
   file7.close();


   return 0;
}
