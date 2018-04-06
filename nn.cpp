#include <iostream>
#include <fstream>
#include <assert.h>
#include <cmath>
#include <Eigen/Dense>
#include "nn.h"

using namespace Eigen;

namespace nnet
{
    neural_net::neural_net(const VectorXi& topology) 
    {
        assert(topology.size()>1);
        init_layers(topology);
        init_weights(0.5);
        autoscale_reset();
    }

    neural_net::neural_net(const char* filename) 
    {
        std::ifstream file(filename, std::ios::in);
        if(file) 
        { 
            // number of layers
            int num_layers;
            file >> num_layers; 
            VectorXi topology(num_layers);
            
            // topology
            for(int i = 0; i < topology.size(); ++i)
                file >> topology[i];
            
            init_layers(topology);
            autoscale_reset();
            
            // scaling parameters
            for(int i = 0; i < x_scale_.size(); ++i) file >> x_scale_[i];
            for(int i = 0; i < x_shift_.size(); ++i) file >> x_shift_[i];
            for(int i = 0; i < y_scale_.size(); ++i) file >> y_scale_[i];
            for(int i = 0; i < y_shift_.size(); ++i) file >> y_shift_[i];
            
            // weights
            for (int i = 1; i < layers_.size(); ++i) 
            {
                auto& layer = layers_[i];
                for(int j = 0; j < layer.b.size(); ++j) file >> layer.b(j);
                for(int j = 0; j < layer.W.size(); ++j) file >> layer.W(j);
            }
        }
        
        file.close();
    }

    void neural_net::init_layers(const VectorXi& topology) 
    {
        // init input layer
        nparam_ = 0; 
        nn_layer l;
        l.size = topology(0);
		for (int i = 0; i < l.size; ++i)
		{
			matrix_t dai;
			(l.da).push_back(dai);		
		}
        layers_.push_back(l);

        // init hidden and output layer
        for (int i = 1; i < topology.size(); ++i) 
        {
            nn_layer l;
            l.size  = topology(i);    
            l.W.setZero(l.size, layers_[i-1].size);    
            l.b.setZero(l.size);
			for (int j = 0; j < layers_.back().size; ++j)
			{
				matrix_t daj;
				(l.da).push_back(daj);		
			}
            layers_.push_back(l);
            nparam_ += l.W.size() + l.b.size();
        }
        tparams_ = {0.005, 1.e10, 10.0, 1.e-7, 0, 1000};

    }

    void neural_net::init_weights(f_type sd) 
    {
        for (int i = 1; i < layers_.size(); ++i) 
        {
            layers_[i].W.setRandom();
            layers_[i].b.setRandom();
            layers_[i].W *= sd;
            layers_[i].b *= sd;
        }
    }

    void neural_net::forward_pass(const matrix_t& X) 
    {
        assert(layers_.front().size == X.cols());
        
        // copy and scale data matrix
        layers_[0].a.noalias() = (X.rowwise() - x_shift_.transpose())*x_scale_.asDiagonal();
				
		// input layer has const. 1st derivative
		for (int i = 0; i < layers_.front().size; ++i)
		{
			(layers_[0]).da[i].noalias() = matrix_t::Zero(X.rows(),X.cols());
			(layers_[0]).da[i].col(i) = vector_t::LinSpaced(X.rows(),1,1);	
		}

        for (int i = 1; i < layers_.size(); ++i)
        {
            // compute input for current layer
            layers_[i].z.noalias() = layers_[i-1].a * layers_[i].W.transpose();
            
            // add bias
            layers_[i].z.rowwise() += layers_[i].b.transpose(); 
            
            // apply activation function
            bool end = (i >= layers_.size() - 1);
            layers_[i].a = end ? layers_[i].z : activation(layers_[i].z);

			// forward propogate derivative
			for (int j = 0; j < layers_.front().size; ++j)
			{
				if(end)
					(layers_[i]).da[j].noalias() = ((layers_[i-1]).da[j] * layers_[i].W.transpose());
				else
					(layers_[i]).da[j].noalias() = (((((layers_[i-1]).da[j] * layers_[i].W.transpose()).array()) * (activation_gradient(layers_[i].a)).array()).matrix());
			}
        }
    }

    f_type neural_net::loss(const matrix_t& X, const matrix_t& Y)
    {
        assert(layers_.front().size == X.cols());
        assert(layers_.back().size == Y.cols());
        assert(X.rows() == Y.rows());
        
        // number of samples and output dim. 
        size_t Q = Y.rows();
        size_t S = Y.cols();
        
        // Resize jacobian and define error. 
        je_.resize(nparam_);
        j_.resize(S*Q, nparam_);
		jj_ = j_.transpose()*j_;
        vector_t error(S*Q);
        
        // MSE. 
        f_type mse = 0.;
		
		// forward pass
		forward_pass(X);
		
		// compute error 
		error = (layers_.back().a*y_scale_.asDiagonal().inverse()).rowwise() + y_shift_.transpose() - Y;
		
		// compute loss
		mse += error.transpose().rowwise().squaredNorm().mean()/S;
		
		// Number of layers. 
        size_t m = layers_.size();
		
		layers_[m-1].delta = error;
		
		size_t j = 0 ;
		
		for(size_t i = layers_.size() - 1; i > 0; --i)
		{	
			layers_[i].dEdW = (layers_[i-1].a.transpose() * layers_[i].delta).transpose();
			layers_[i-1].delta = ((layers_[i].delta * layers_[i].W).array() * activation_gradient(layers_[i-1].a).array()).matrix();
			je_.segment(j,layers_[i].W.size()) = Map<vector_t>(layers_[i].dEdW.data(),layers_[i].dEdW.size());
			std::cout << je_ << std::endl;
			std::cout << std::endl;
			j += layers_[i].W.size();
			je_.segment(j,layers_[i].b.size()) = layers_[i].delta.colwise().sum();
			j += layers_[i].b.size();
			std::cout << je_ << std::endl;
			std::cout << std::endl;
		}
		
		
		std::cout << je_ << std::endl;
		std::cout << std::endl;
		vector_t temp = je_ / (Q*S);
        
		std::cout << temp << std::endl;
		std::cout << std::endl;
	
		std::cout << (layers_[3].dEdW).array() / (Q*S) <<std::endl; 
		std::cout << std::endl;
		
        for(size_t k = 0; k < Q; ++k)
        {
            // forward pass
            forward_pass(X.row(k));
                
            // compute error
			error.segment(k*S, S) = (layers_.back().a*y_scale_.asDiagonal().inverse()).transpose() - (Y.row(k).transpose() - y_shift_);
            
            // Compute loss. 
            mse += error.segment(k*S, S).transpose().rowwise().squaredNorm().mean()/S;
            
            // Number of layers. 
            size_t m = layers_.size();

            // Compute sensitivities. 
            size_t j = nparam_;
            layers_[m-1].delta = y_scale_.asDiagonal().inverse();
            
            // Pack Jacobian.
            j -= layers_[m-1].W.size();
            for(size_t p = 0; p < S; ++p)
            {
                layers_[m-1].dEdW = (layers_[m-1].delta.col(p)*layers_[m-2].a);
                j_.block(S*k+p, j, 1, layers_[m-1].W.size()) = Map<vector_t>(layers_[m-1].dEdW.data(), layers_[m-1].dEdW.size()).transpose();
            }
        
            j -= layers_[m-1].b.size();
            j_.block(S*k, j, S, layers_[m-1].b.size()) = layers_[m-1].delta;

            for(size_t i = layers_.size() - 2; i > 0; --i)
            {
                layers_[i].delta.noalias() = activation_gradient(layers_[i].a).asDiagonal()*layers_[i+1].W.transpose()*layers_[i+1].delta;

                // Pack Jacobian.
                j -= layers_[i].W.size();
                for(size_t p = 0; p < S; ++p)
                {
                    layers_[i].dEdW.noalias() = (layers_[i].delta.col(p)*layers_[i-1].a);            
                    j_.block(S*k+p, j, 1, layers_[i].W.size()) = Map<vector_t>(layers_[i].dEdW.data(), layers_[i].dEdW.size()).transpose();
                }

                j -= layers_[i].b.size();
                j_.block(S*k, j, S, layers_[i].b.size()) = layers_[i].delta.transpose();
            }
        }
		
		//std::cout << j_.rows()<< " ; " << j_.cols() << std::endl;
		

        jj_.noalias() = j_.transpose()*j_;
		//std::cout << jj_.rows()<< " ; " << jj_.cols() << std::endl;
        jj_ /= (Q*S);
        j_ /= (Q*S);
        je_.noalias() = j_.transpose()*error;
		std::cout << je_  << std::endl;

		//std::cout << je_.rows()<< " ; " << je_.cols() << std::endl;
		exit(1);
        return mse/Q;
		
    }

	f_type neural_net::loss(const matrix_t& X, const matrix_t& Y, const std::vector<matrix_t> &Z, double ratio)
    {
        assert(layers_.front().size == X.cols());
        assert(layers_.back().size == Y.cols());
        assert(X.rows() == Y.rows());
        
        // number of samples and output dim. 
        size_t Q = Y.rows();
        size_t S = Y.cols();
        
        // Resize jacobian and define error. 
        je_.resize(nparam_);
        j_.resize(S*Q, nparam_);
        vector_t error(S*Q);

		// Define error on d(output)/d(input) gradient
		std::vector <vector_t> dererror;
		for(size_t i = 0; i < X.cols(); ++i)
			dererror.push_back(error);		
        
        // MSE. 
        f_type mse = 0.;
        
		// Loop over samples.
        for(size_t k = 0; k < Q; ++k)
        {
            // forward pass
            forward_pass(X.row(k));

			std::cout << X.row(k) << std::endl;
			exit(1);
                
            // compute error
			error.segment(k*S, S) = (layers_.back().a*y_scale_.asDiagonal().inverse()).transpose() - (Y.row(k).transpose() - y_shift_);

			// compute error on derivative
			for(size_t i = 0; i < X.cols(); ++i)
				dererror[i].segment(k*S, S) = (layers_.back().da[i]*y_scale_.asDiagonal().inverse()).transpose()*x_scale_.row(i) - (Z[i].row(k).transpose());
            
            // Compute loss. 
            mse += error.segment(k*S, S).transpose().rowwise().squaredNorm().mean()/S*ratio;
			for(size_t i = 0; i < X.cols(); ++i)
				mse += dererror[i].segment(k*S, S).transpose().rowwise().squaredNorm().mean()/S*(1.0/ratio);
            
            // Number of layers. 
            size_t m = layers_.size();

            // Compute sensitivities. 
            size_t j = nparam_;
            layers_[m-1].delta = y_scale_.asDiagonal().inverse();
            
            // Pack Jacobian.
            j -= layers_[m-1].W.size();
            for(size_t p = 0; p < S; ++p)
            {
                layers_[m-1].dEdW.noalias() = (layers_[m-1].delta.col(p)*layers_[m-2].a);
                j_.block(S*k+p, j, 1, layers_[m-1].W.size()) = Map<vector_t>(layers_[m-1].dEdW.data(), layers_[m-1].dEdW.size()).transpose();
            }
        
            j -= layers_[m-1].b.size();
            j_.block(S*k, j, S, layers_[m-1].b.size()) = layers_[m-1].delta;

            for(size_t i = layers_.size() - 2; i > 0; --i)
            {
                layers_[i].delta.noalias() = activation_gradient(layers_[i].a).asDiagonal()*layers_[i+1].W.transpose()*layers_[i+1].delta;

                // Pack Jacobian.
                j -= layers_[i].W.size();
                for(size_t p = 0; p < S; ++p)
                {
                    layers_[i].dEdW.noalias() = (layers_[i].delta.col(p)*layers_[i-1].a);            
                    j_.block(S*k+p, j, 1, layers_[i].W.size()) = Map<vector_t>(layers_[i].dEdW.data(), layers_[i].dEdW.size()).transpose();
                }

                j -= layers_[i].b.size();
                j_.block(S*k, j, S, layers_[i].b.size()) = layers_[i].delta.transpose();
            }
        }

        jj_.noalias() = j_.transpose()*j_;
        jj_ /= (Q*S);
        j_ /= (Q*S);
        je_.noalias() = j_.transpose()*error;
        return mse/Q;
    }

    void neural_net::train(const matrix_t& X, const matrix_t& Y, const std::vector<matrix_t> &Z, double ratio, bool verbose)
    {
        // Reset mu.
        tparams_.mu = 0.005;

        int nex = X.rows();
        
        // Forward and back propogate to compute loss and Jacobian.
        f_type mse = loss(X, Y);
        vector_t wb = get_wb(), optwb = get_wb();
        f_type wse = wb.transpose()*wb;

        // Initialize Bayesian regularization parameters.
        f_type gamma = nparam_;
        f_type beta = 0.5*(nex - gamma)/mse;
        beta = beta <= 0 ? 1. : beta;
        f_type alpha = 0.5*gamma/wse;
        f_type tse = beta*mse + alpha*wse;
        f_type grad = 2.*std::sqrt(je_.squaredNorm());
        matrix_t eye = matrix_t::Identity(nparam_, nparam_);
        
        // Define iteration tol parameters.
        int iter = 0;
        f_type tse2 = 0, mse2 = 0, wse2 = 0;
        do
        {
            if(verbose)
                std::cout << "iter: " << iter << " mse: " << mse << " gamma: " << gamma << " mu: " << tparams_.mu << " grad: " << grad << std::endl;
            
            matrix_t jjb = jj_; 
            vector_t jeb = je_;
            do
            {
                // Compute new weights and performance.
                optwb = wb - (beta*jjb + (tparams_.mu + alpha)*eye).colPivHouseholderQr().solve(beta*jeb + alpha*wb);
                wse2 = optwb.transpose()*optwb;
                set_wb(optwb);
                mse2 = loss(X, Y);
                tse2 = beta*mse2 + alpha*wse2;
                
                // Exit loop or reset values.
                if(tse2 < tse || tparams_.mu > tparams_.mu_max)
                    break;
                else
                {
                    set_wb(wb);
                    //mse2 = loss(X, Y);
                    tparams_.mu *= tparams_.mu_scale;           
                }
            } while(true);

            wb = optwb;
            mse = mse2; wse = wse2;
            gamma = (f_type)nparam_ - alpha*(beta*jj_ + alpha*eye).inverse().trace();
            beta = mse == 0 ? 1. : 0.5*((f_type)nex - gamma)/mse;
            alpha = wse == 0 ? 1. : 0.5*gamma/wse;
            tse = beta*mse + alpha*wse;
            grad = 2.*std::sqrt(je_.squaredNorm());
            
            if(tparams_.mu < tparams_.mu_max)
                tparams_.mu /= tparams_.mu_scale;
            if(tparams_.mu < 1.e-20) tparams_.mu = 1.e-20;

            ++iter;        
        } while(
            tparams_.mu < tparams_.mu_max && 
            grad > tparams_.min_grad && 
            iter <= tparams_.max_iter && 
            mse > tparams_.min_loss &&
            !std::isnan(grad) && 
            !std::isnan(gamma));
    
        if(verbose)
            std::cout << "iter: " << iter << " mse: " << mse << " gamma: " << gamma << " mu: " << tparams_.mu << " grad: " << grad << std::endl;
    }

	void neural_net::train(const matrix_t& X, const matrix_t& Y, bool verbose)
    {
        // Reset mu.
        tparams_.mu = 0.005;

        int nex = X.rows();
        
        // Forward and back propogate to compute loss and Jacobian.
        f_type mse = loss(X, Y);
        vector_t wb = get_wb(), optwb = get_wb();
        f_type wse = wb.transpose()*wb;

        // Initialize Bayesian regularization parameters.
        f_type gamma = nparam_;
        f_type beta = 0.5*(nex - gamma)/mse;
        beta = beta <= 0 ? 1. : beta;
        f_type alpha = 0.5*gamma/wse;
        f_type tse = beta*mse + alpha*wse;
        f_type grad = 2.*std::sqrt(je_.squaredNorm());
        matrix_t eye = matrix_t::Identity(nparam_, nparam_);
        
        // Define iteration tol parameters.
        int iter = 0;
        f_type tse2 = 0, mse2 = 0, wse2 = 0;
        do
        {
            if(verbose)
                std::cout << "iter: " << iter << " mse: " << mse << " gamma: " << gamma << " mu: " << tparams_.mu << " grad: " << grad << std::endl;
            
            matrix_t jjb = jj_; 
            vector_t jeb = je_;
            do
            {
                // Compute new weights and performance.
                optwb = wb - (beta*jjb + (tparams_.mu + alpha)*eye).colPivHouseholderQr().solve(beta*jeb + alpha*wb);
                wse2 = optwb.transpose()*optwb;
                set_wb(optwb);
                mse2 = loss(X, Y);
                tse2 = beta*mse2 + alpha*wse2;
                
                // Exit loop or reset values.
                if(tse2 < tse || tparams_.mu > tparams_.mu_max)
                    break;
                else
                {
                    set_wb(wb);
                    //mse2 = loss(X, Y);
                    tparams_.mu *= tparams_.mu_scale;           
                }
            } while(true);

            wb = optwb;
            mse = mse2; wse = wse2;
            gamma = (f_type)nparam_ - alpha*(beta*jj_ + alpha*eye).inverse().trace();
            beta = mse == 0 ? 1. : 0.5*((f_type)nex - gamma)/mse;
            alpha = wse == 0 ? 1. : 0.5*gamma/wse;
            tse = beta*mse + alpha*wse;
            grad = 2.*std::sqrt(je_.squaredNorm());
            
            if(tparams_.mu < tparams_.mu_max)
                tparams_.mu /= tparams_.mu_scale;
            if(tparams_.mu < 1.e-20) tparams_.mu = 1.e-20;

            ++iter;        
        } while(
            tparams_.mu < tparams_.mu_max && 
            grad > tparams_.min_grad && 
            iter <= tparams_.max_iter && 
            mse > tparams_.min_loss &&
            !std::isnan(grad) && 
            !std::isnan(gamma));
    
        if(verbose)
            std::cout << "iter: " << iter << " mse: " << mse << " gamma: " << gamma << " mu: " << tparams_.mu << " grad: " << grad << std::endl;
    }

    train_param neural_net::get_train_params() const
    {
        return tparams_;
    }

    void neural_net::set_train_params(const train_param& params)
    {
        tparams_ = params;
    }

    matrix_t neural_net::get_activation() 
    {
        return (layers_.back().a*y_scale_.asDiagonal().inverse()).rowwise() + y_shift_.transpose();
    }

	matrix_t neural_net::get_gradient_forwardpass(int index) 
    {
        return ((layers_.back().da[index])*y_scale_.asDiagonal().inverse())*x_scale_.row(index);
    }

    matrix_t neural_net::get_gradient(int index)
    {
        layers_.back().delta = matrix_t::Identity(layers_.back().size, layers_.back().size)*y_scale_.asDiagonal().inverse();
        for (size_t i = layers_.size() - 2; i > 0; --i) 
        {
            matrix_t g = activation_gradient(layers_[i].a);
            layers_[i].delta = (layers_[i+1].delta*layers_[i+1].W)*g.row(index).asDiagonal();
        }

        return layers_[1].delta*layers_[1].W*x_scale_.asDiagonal();
    }


	// this is tanh(x)
    matrix_t neural_net::activation(const matrix_t& x)  
    {
        return (2.*((-2.*x).array().exp() + 1.0).inverse() - 1.0).matrix();
    }


	// derivative is 1-tanh^2(x) = sech^2
    matrix_t neural_net::activation_gradient(const matrix_t& x) 
    {
        return (1.0-x.array().square()).matrix();
    }

	// 2nd derivative is -2*tanh*sech^2 = -2*x*(1-x^2) 
    matrix_t neural_net::activation_secondgradient(const matrix_t& x) 
    {
        return (-2.*(1.0-x.array().square())*(x.array())).matrix();
    }

    void neural_net::set_wb(const vector_t& wb)
    {
        int k = 0;
        for (int i = 1; i < layers_.size(); ++i) 
        {
            auto& layer = layers_[i];
            for(int j = 0; j < layer.b.size(); ++j)
            {
                layer.b(j) = wb[k];
                ++k;
            }

            for(int j = 0; j < layer.W.size(); ++j)
            {
                layer.W(j) = wb[k];
                ++k;
            }
        }
    }

    vector_t neural_net::get_wb() const
    {
        int k = 0;
        vector_t wb(nparam_);
        for (int i = 1; i < layers_.size(); ++i) 
        {
            auto& layer = layers_[i];
            for(int j = 0; j < layer.b.size(); ++j)
            {
                wb[k] = layer.b(j);
                ++k;
            }

            for(int j = 0; j < layer.W.size(); ++j)
            {
                wb[k] = layer.W(j);
                ++k;
            }
        }

        return wb;
    }

    void neural_net::autoscale(const matrix_t& X, const matrix_t& Y) 
    {
        assert(layers_.front().size == X.cols());
        assert(layers_.back().size == Y.cols());
        assert(X.rows() == Y.rows());
        
        x_shift_ = 0.5*(X.colwise().minCoeff().array() + X.colwise().maxCoeff().array());
        x_scale_ = 2.0*(X.colwise().maxCoeff() - X.colwise().minCoeff()).array().inverse();

        y_shift_ = 0.5*(Y.colwise().minCoeff().array() + Y.colwise().maxCoeff().array());
        y_scale_ = 2.0*(Y.colwise().maxCoeff() - Y.colwise().minCoeff()).array().inverse();
    }

    void neural_net::autoscale_reset() 
    {
        x_scale_ = vector_t::Ones(layers_.front().size);
        x_shift_ = vector_t::Zero(layers_.front().size);
        y_scale_ = vector_t::Ones(layers_.back().size);
        y_shift_ = vector_t::Zero(layers_.back().size);
    }

    bool neural_net::write(const char* filename) 
    {
        // open file
        std::ofstream file(filename, std::ios::out);
        
        // write everything to disk
        if(file) 
        {
            file.precision(16);
            file << std::scientific;

            // number of layers
            file << static_cast<int>(layers_.size()) << std::endl;

            // topology
            for (int i = 0; i < layers_.size() - 1; ++i) 
                file << static_cast<int>(layers_[i].size) << " ";
            file << static_cast<int>(layers_.back().size) << std::endl;

            for(int i = 0; i < x_scale_.size() - 1; ++i) file << x_scale_[i] << " ";
            file << x_scale_[x_scale_.size()-1] << std::endl;

            for(int i = 0; i < x_shift_.size() - 1; ++i) file << x_shift_[i] << " ";
            file << x_shift_[x_shift_.size()-1] << std::endl;

            for(int i = 0; i < y_scale_.size() - 1; ++i) file << y_scale_[i] << " ";
            file << y_scale_[y_scale_.size()-1] << std::endl;

            for(int i = 0; i < y_shift_.size() - 1; ++i) file << y_shift_[i] << " ";
            file << y_shift_[y_shift_.size()-1] << std::endl;
            
            // weights
            for (int i = 1; i < layers_.size(); ++i) 
            {
                auto& layer = layers_[i];
                for(int j = 0; j < layer.b.size(); ++j) file << layer.b(j) << std::endl;
                for(int j = 0; j < layer.W.size(); ++j) file << layer.W(j) << std::endl;
            }
        } 
        else
            return false;
        
        file.close();
        return true;
    }

    neural_net::~neural_net() 
    {
    }
}
