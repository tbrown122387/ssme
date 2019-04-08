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

    
    //! get number of parameters
    /**
     * @brief gets the number of parameters in your pack
     * @return an unsigned integer 
     */
    unsigned int getNumParams() const;


    //! get the transformed parameters in the unrestricted space
    /**
     * @brief get the transformed parameters on the unrestricted space
     * @return an Eigen::Vector of transformed parameters
     */
    Eigen::VectorXd getTransParams() const;


    //! get the untransformed parameters in the possibly-restricted space
    /**
     * @brief get the untransformed parameters on the possibly-restricted space
     * @return an Eigen::Vector of transformed parameters
     */
    Eigen::VectorXd getUnTransParams() const;


    //! get a subset of the transformed parameters in the unrestricted space
    // example: pp.getTransParams(0,2) returns three elements
    /**
     * @brief get a subset of the transformed parameters on the unrestricted space
     * @param index of first element (starts counting at zero)
     * @param index of last element (not like python indexing!)
     * @return an Eigen::Vector of transformed parameters
     */ 
    Eigen::VectorXd getTransParams(const unsigned int& start, const unsigned int& end) const;


    //! get a subset of the untransformed parameters in the possibly-restricted space
    // example: pp.getTransParams(0,2) returns three elements
    /**
    * @brief get a subset of the transformed parameters on the unrestricted space
    * @param index of first element (starts counting at zero)
    * @param index of last element (not like python indexing!)
    * @return an Eigen::Vector of transformed parameters
    */ 
    Eigen::VectorXd getUnTransParams(const unsigned int& start, const unsigned int& end) const;
    
    
    //! get the log of the Jacobian determinant you need for the density of transformed parameters.
    /**
     * @brief get the log of the Jacobian determinant you need for the density of transformed params
     * @return a double
     */
    double getLogJacobian() const;
    

    //! copy values from another paramPack
    /**
     * @brief copy the values of another paramPack
     * @param the other parameter pack whose values you want to take as your own
     */
    void takeValues(const paramPack& other); 


private:
    
    Eigen::VectorXd m_trans_params;
    vecptrs m_transform_functors;
    
};

#endif // PARAM_PACK_H
