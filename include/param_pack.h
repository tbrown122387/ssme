#ifndef PARAM_PACK_H
#define PARAM_PACK_H

#include <Eigen/Dense>
#include <vector>
#include <memory> // std::unique_ptr

#include "param_transforms.h"


/**
 * @class paramPack
 * @author t
 * @file param_pack.h
 * @brief Stores transformed parameters, as well as the functions 
 * that can change them to the untransformed parameters and back.
 */
class paramPack{
public:

    // type aliases
    using vecptrs = std::vector<std::unique_ptr<paramTransform>>;

    // ctors
    paramPack(const paramPack& pp) = delete; // disallow copy constructor
    paramPack(const Eigen::VectorXd &trans_params, vecptrs&& t_functors);
    paramPack(const Eigen::VectorXd &params, const std::vector<TransType> &vec_trans_types, bool start_w_trans_params = true);
    
    // assignment operators
    paramPack& operator=(const paramPack& other) = delete; // disallow assignment because std::vector<std::unique_ptr is move-only
    paramPack& operator=(paramPack&& other) = delete;

    // getters
    unsigned int getNumParams() const;
    Eigen::VectorXd getTransParams() const;
    Eigen::VectorXd getUnTransParams() const;
    Eigen::VectorXd getTransParams(const unsigned int& start, const unsigned int& end) const;
    Eigen::VectorXd getUnTransParams(const unsigned int& start, const unsigned int& end) const;
    double getLogJacobian() const;
    
    // setters
    void takeValues(const paramPack& other); // TODO: make sure the sizes are the same.


private:
    
    Eigen::VectorXd m_trans_params;
    vecptrs m_transform_functors;
    
};

#endif // PARAM_PACK_H
