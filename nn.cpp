// Copyright (c) 2013, Manuel Blum
// All rights reserved.

#include <iostream>
#include <fstream>
#include <assert.h>

#include "nn.h"

neural_net::neural_net(Eigen::VectorXi& topology) 
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
        Eigen::VectorXi topology(num_layers);
        
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

void neural_net::init_layers(Eigen::VectorXi& topology) 
{
    // init input layer
    nn_layer l;
    l.size = topology(0);
    layers_.push_back(l);
    // init hidden and output layer
    for (int i = 1; i < topology.size(); ++i) 
    {
        nn_layer l;
        l.size  = topology(i);    
        l.W.setZero(l.size, layers_[i-1].size);    
        l.b.setZero(l.size);
        layers_.push_back(l);
    }  
}

void neural_net::init_weights(F_TYPE sd) 
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
    layers_[0].a = (X.rowwise() - x_shift_.transpose())*x_scale_.asDiagonal();
    for (int i = 1; i < layers_.size(); ++i)
    {
        // compute input for current layer
        layers_[i].z = layers_[i-1].a * layers_[i].W.transpose();
        
        // add bias
        layers_[i].z.rowwise() += layers_[i].b.transpose(); 
        
        // apply activation function
        bool end = (i >= layers_.size() - 1);
        layers_[i].a = end ? layers_[i].z : activation(layers_[i].z);
    }
}

matrix_t neural_net::get_activation() 
{
    return (layers_.back().a*y_scale_.asDiagonal().inverse()).rowwise() + y_shift_.transpose();
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

matrix_t neural_net::activation(const matrix_t& x)  
{
    //return ((-x).array().exp() + 1.0).inverse().matrix();
    return x.array().tanh().matrix();
}

matrix_t neural_net::activation_gradient(const matrix_t& x) 
{
    //return x.cwiseProduct((1.0-x.array()).matrix());
    return (1.0-x.array()*x.array()).matrix();
}

void neural_net::autoscale(const matrix_t& X, const matrix_t& Y) 
{
    assert(layers_.front().size == X.cols());
    assert(layers_.back().size == Y.cols());
    assert(X.rows() == Y.rows());
    
    // compute the mean of the input data
    x_shift_ = X.colwise().mean();
    
    // compute the standard deviation of the input data
    x_scale_ = (X.rowwise() - x_shift_.transpose()).array().square().colwise().mean().array().sqrt().inverse();
    for (size_t i = 0; i < x_scale_.size(); ++i) if (x_scale_(i) > 10e9) x_scale_(i) = 1;
    
    // compute the minimum target values
    y_shift_ = Y.colwise().minCoeff();
    
    // compute the maximum shifted target values
    y_scale_ = (Y.colwise().maxCoeff() - y_shift_.transpose()).array().inverse();
    for (size_t i = 0; i < y_scale_.size(); ++i) if (y_scale_(i) > 10e9) y_scale_(i) = 1;
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
    if (file) 
    {
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

