#include "param_pack.h"

#include <utility> // std::move
#include <stdexcept> // std::invalid_argument


paramPack::paramPack(const Eigen::VectorXd& trans_params, vecptrs&& t_functors)
    : m_trans_params(trans_params), m_transform_functors(std::move(t_functors))
{
}


paramPack::paramPack(const Eigen::VectorXd &params, const std::vector<TransType> &vec_trans_types, bool start_w_trans_params)
{
    unsigned int n = vec_trans_types.size();
    if( params.rows() != vec_trans_types.size() ){
        throw std::invalid_argument("params and vec_trans_types have to be the same size");
    }
    
    for(auto & tt : vec_trans_types){
        m_transform_functors.push_back(paramInvTransform::create(tt));
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


unsigned int paramPack::getNumParams() const
{
    return m_transform_functors.size();
}


Eigen::VectorXd paramPack::getTransParams() const
{
    return m_trans_params.block(0,0,this->getNumParams(),1);
}


Eigen::VectorXd paramPack::getUnTransParams() const
{
    unsigned int n = this->getNumParams();
    Eigen::VectorXd params(n);
    for(size_t i = 0; i < n; ++i)
        params(i) = m_transform_functors[i]->invTrans(m_trans_params(i));
    return params;    
}


Eigen::VectorXd paramPack::getTransParams(const unsigned int& start, const unsigned int& end) const
{
    return m_trans_params.block(start,0,(end-start+1),1);
}


Eigen::VectorXd paramPack::getUnTransParams(const unsigned int& start, const unsigned int& end) const
{    
    Eigen::VectorXd params(end-start+1);
    for(size_t i = start; i < end + 1; ++i)
        params(i-start) = m_transform_functors[i]->invTrans(m_trans_params(i));
    return params;
}


double paramPack::getLogJacobian() const
{
    double result(0.0);
    for(size_t i = 0; i < m_transform_functors.size(); ++i){
        result += m_transform_functors[i]->logJacobian(m_trans_params(i));
    }
    return result;
}


void paramPack::takeValues(const paramPack& other)
{
    if(m_transform_functors.size() != other.getNumParams())
        throw std::invalid_argument("other must have the same size as the caller");
    m_trans_params = other.getTransParams(0,m_trans_params.size()-1); // TODO: what happens when they are of different sizes?!
}
