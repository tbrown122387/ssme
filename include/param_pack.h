#ifndef PARAM_PACK_H
#define PARAM_PACK_H

#include <Eigen/Dense>
#include <vector>
#include <memory> // std::unique_ptr
#include <utility> // std::move
#include <stdexcept> // std::invalid_argument

#include "param_transforms.h"


/**
 * @class paramPack
 * @author t
 * @file param_pack.h
 * @brief Stores transformed parameters, as well as the functions 
 * that can change them to the untransformed parameters and back.
 */
template<typename float_t>
class paramPack{
public:

    // type aliases
    using vecptrs = std::vector<std::unique_ptr<paramTransform<float_t>>>;
    using eig_dyn_vec = Eigen::Matrix<float_t,Eigen::Dynamic,1>;

    // ctors
    paramPack(const paramPack<float_t>& pp) = delete; // disallow copy constructor
    paramPack(const eig_dyn_vec &trans_params, vecptrs&& t_functors);
    paramPack(const eig_dyn_vec &params, const std::vector<TransType> &vec_trans_types, bool start_w_trans_params = true);
    
    // assignment operators
    paramPack& operator=(const paramPack<float_t>& other) = delete; // disallow assignment because std::vector<std::unique_ptr is move-only
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
    auto getTransParams() const -> eig_dyn_vec;


    //! get the untransformed parameters in the possibly-restricted space
    /**
     * @brief get the untransformed parameters on the possibly-restricted space
     * @return an Eigen::Vector of transformed parameters
     */
    auto getUnTransParams() const -> eig_dyn_vec;


    //! get a subset of the transformed parameters in the unrestricted space
    // example: pp.getTransParams(0,2) returns three elements
    /**
     * @brief get a subset of the transformed parameters on the unrestricted space
     * @param index of first element (starts counting at zero)
     * @param index of last element (not like python indexing!)
     * @return an Eigen::Vector of transformed parameters
     */ 
    auto getTransParams(const unsigned int& start, const unsigned int& end) const -> eig_dyn_vec;


    //! get a subset of the untransformed parameters in the possibly-restricted space
    // example: pp.getTransParams(0,2) returns three elements
    /**
    * @brief get a subset of the transformed parameters on the unrestricted space
    * @param index of first element (starts counting at zero)
    * @param index of last element (not like python indexing!)
    * @return an Eigen::Vector of transformed parameters
    */ 
    auto getUnTransParams(const unsigned int& start, const unsigned int& end) const -> eig_dyn_vec;
    
    
    //! get the log of the Jacobian determinant you need for the density of transformed parameters.
    /**
     * @brief get the log of the Jacobian determinant you need for the density of transformed params
     * @return a float_t
     */
    auto getLogJacobian() const -> float_t;
    

    //! copy values from another paramPack
    /**
     * @brief copy the values of another paramPack
     * @param the other parameter pack whose values you want to take as your own
     */
    void takeValues(const paramPack<float_t>& other); 


private:
    
    eig_dyn_vec m_trans_params;
    vecptrs m_transform_functors;
    
};


template<typename float_t>
paramPack<float_t>::paramPack(const eig_dyn_vec& trans_params, vecptrs&& t_functors)
    : m_trans_params(trans_params), m_transform_functors(std::move(t_functors))
{
}


template<typename float_t>
paramPack<float_t>::paramPack(const eig_dyn_vec &params, const std::vector<TransType> &vec_trans_types, bool start_w_trans_params)
{
    unsigned int n = vec_trans_types.size();
    if( params.rows() != vec_trans_types.size() ){
        throw std::invalid_argument("params and vec_trans_types have to be the same size");
    }
    
    for(auto & tt : vec_trans_types){
        m_transform_functors.push_back(paramTransform<float_t>::create(tt));
    }
    
    if(start_w_trans_params){
        m_trans_params = params;
    }else if(!start_w_trans_params){
        m_trans_params.resize(n);
        for(size_t i = 0; i < n; ++i)
            m_trans_params(i) = m_transform_functors[i]->trans(params(i));            
            
    }else{
        throw std::invalid_argument("your vec_trans_types argument is the wrong size!");
    }
}


template<typename float_t>
unsigned int paramPack<float_t>::getNumParams() const
{
    return m_transform_functors.size();
}


template<typename float_t>
auto paramPack<float_t>::getTransParams() const -> eig_dyn_vec 
{
    return m_trans_params.block(0,0,this->getNumParams(),1);
}


template<typename float_t>
auto paramPack<float_t>::getUnTransParams() const -> eig_dyn_vec
{
    unsigned int n = this->getNumParams();
    eig_dyn_vec params(n);
    for(size_t i = 0; i < n; ++i)
        params(i) = m_transform_functors[i]->invTrans(m_trans_params(i));
    return params;    
}


template<typename float_t>
auto paramPack<float_t>::getTransParams(const unsigned int& start, const unsigned int& end) const -> eig_dyn_vec
{
    return m_trans_params.block(start,0,(end-start+1),1);
}


template<typename float_t>
auto paramPack<float_t>::getUnTransParams(const unsigned int& start, const unsigned int& end) const -> eig_dyn_vec
{    
    eig_dyn_vec params(end-start+1);
    for(size_t i = start; i < end + 1; ++i)
        params(i-start) = m_transform_functors[i]->invTrans(m_trans_params(i));
    return params;
}


template<typename float_t>
float_t paramPack<float_t>::getLogJacobian() const
{
    float_t result(0.0);
    for(size_t i = 0; i < m_transform_functors.size(); ++i){
        result += m_transform_functors[i]->logJacobian(m_trans_params(i));
    }
    return result;
}


template<typename float_t>
void paramPack<float_t>::takeValues(const paramPack<float_t>& other)
{
    if(m_transform_functors.size() != other.getNumParams())
        throw std::invalid_argument("other must have the same size as the caller");
    m_trans_params = other.getTransParams(0,m_trans_params.size()-1); // TODO: what happens when they are of different sizes?!
}




#endif // PARAM_PACK_H
