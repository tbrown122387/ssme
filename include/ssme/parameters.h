#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <memory> // unique_ptr
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include <utility> // std::move
#include <type_traits>



/**
 * @namespace param
 * @brief for parameter-related things such as 
 * transforming parameters and collecting them.
 * @file parameters.h
 */
namespace param {


/** enum class for different types of transformations */
enum class trans_type {TT_null, TT_twice_fisher, TT_logit, TT_log};


/**
 * @class transform
 * @author Taylor
 * @brief a pure virtual base class. cts params only.
 */
template<typename float_t>
class transform{

    static_assert(std::is_floating_point<float_t>::value, "must be floating point type");

public:

    /**
     * @brief a virtual destructor
     */
    virtual ~transform() {}


    /**
     * @brief from the constrained/nontransformed to the transformed/unconstrained space
     */
    virtual float_t trans(const float_t& p) = 0;


    /**
     * @brief from the unconstrained/transformed to the constrained/untransformed space;
     */
    virtual float_t inv_trans(const float_t &trans_p) = 0;

    /**
     * @brief get the log jacobian (from untransformed to transformed). For use with adjusting priors to the transformed space.
     */
    virtual float_t log_jacobian(const float_t &p) = 0;


    /**
     * @brief a static method to create unique pointers 
     */
    static std::unique_ptr<transform<float_t> > create(trans_type tt);
};


/**
 * @class null_trans
 * @author Taylor
 * @brief trans_p = orig_p.
 */
template<typename float_t>
class null_trans : public transform<float_t>{
public: 
    float_t trans(const float_t& p) override;
    float_t inv_trans(const float_t& trans_p) override;
    float_t log_jacobian(const float_t& p) override;
};


/**
 * @class twice_fisher_trans
 * @author Taylor
 * @brief trans_p = log(1+orig_p) - log(1-orig_p) = logit((orig_p+1)/2).
 */
template<typename float_t>
class twice_fisher_trans : public transform<float_t>{
public:
    float_t trans(const float_t& p) override;
    float_t inv_trans(const float_t &trans_p) override;   
    float_t log_jacobian(const float_t &p) override;
};


/**
 * @class logit_trans
 * @author Taylor
 * @brief trans_p = logit(orig_p).
 */
template<typename float_t>
class logit_trans : public transform<float_t>{
public:
    float_t trans(const float_t& p) override;
    float_t inv_trans(const float_t &trans_p) override;
    float_t log_jacobian(const float_t &p) override;
};


/**
 * @class log_trans
 * @author Taylor
 * @brief trans_p = log(orig_p).
 */
template<typename float_t>
class log_trans : public transform<float_t>{
public:
    float_t trans(const float_t& p) override;
    float_t inv_trans(const float_t &trans_p) override;
    float_t log_jacobian(const float_t &p) override;
};


/**
 * @class pack
 * @author Taylor
 * @brief Stores transformed parameters, as well as the functions 
 * that can change them to the untransformed parameters and back.
 */
template<typename float_t>
class pack{
public:

    // type aliases
    using vecptrs = std::vector<std::unique_ptr<transform<float_t>>>;
    using eig_dyn_vec = Eigen::Matrix<float_t,Eigen::Dynamic,1>;

    // ctors
    pack(const pack<float_t>& pp) = delete; // disallow copy constructor
    pack(const eig_dyn_vec &trans_params, vecptrs&& t_functors);
    pack(const eig_dyn_vec &params, const std::vector<trans_type> &vec_trans_types, bool start_w_trans_params = true);
    
    // assignment operators
    pack& operator=(const pack<float_t>& other) = delete; // disallow assignment because std::vector<std::unique_ptr is move-only
    pack& operator=(pack&& other) = delete;

    
    //! get number of parameters
    /**
     * @brief gets the number of parameters in your pack
     * @return an unsigned integer 
     */
    unsigned int get_num_params() const;


    //! get the transformed parameters in the unrestricted space
    /**
     * @brief get the transformed parameters on the unrestricted space
     * @return an Eigen::Vector of transformed parameters
     */
    auto get_trans_params() const -> eig_dyn_vec;


    //! get the untransformed parameters in the possibly-restricted space
    /**
     * @brief get the untransformed parameters on the possibly-restricted space
     * @return an Eigen::Vector of transformed parameters
     */
    auto get_untrans_params() const -> eig_dyn_vec;


    //! get a subset of the transformed parameters in the unrestricted space
    // example: pp.get_trans_params(0,2) returns three elements
    /**
     * @brief get a subset of the transformed parameters on the unrestricted space
     * @param index of first element (starts counting at zero)
     * @param index of last element (not like python indexing!)
     * @return an Eigen::Vector of transformed parameters
     */ 
    auto get_trans_params(const unsigned int& start, const unsigned int& end) const -> eig_dyn_vec;


    //! get a subset of the untransformed parameters in the possibly-restricted space
    // example: pp.get_trans_params(0,2) returns three elements
    /**
    * @brief get a subset of the transformed parameters on the unrestricted space
    * @param index of first element (starts counting at zero)
    * @param index of last element (not like python indexing!)
    * @return an Eigen::Vector of transformed parameters
    */ 
    auto get_untrans_params(const unsigned int& start, const unsigned int& end) const -> eig_dyn_vec;
    
    
    //! get the log of the Jacobian determinant you need for the density of transformed parameters.
    /**
     * @brief get the log of the Jacobian determinant you need for the density of transformed params
     * @return a float_t
     */
    auto get_log_jacobian() const -> float_t;
    

    //! copy values from another pack
    /**
     * @brief copy the values of another pack
     * @param the other parameter pack whose values you want to take as your own
     */
    void take_values(const pack<float_t>& other); 


private:
    
    eig_dyn_vec m_trans_params;
    vecptrs m_transform_functors;
    
};


////////////////////////////////////////////////////////////////////////////////////


template<typename float_t>
std::unique_ptr<transform<float_t> > transform<float_t>::create(trans_type tt)
{
    if(tt == trans_type::TT_null){
        
        return std::unique_ptr<transform<float_t> >(new null_trans<float_t> );
    
    }else if(tt == trans_type::TT_twice_fisher){
        
        return std::unique_ptr<transform<float_t> >(new twice_fisher_trans<float_t> );
    
    }else if(tt == trans_type::TT_logit){
        
        return std::unique_ptr<transform<float_t> >(new logit_trans<float_t> );
    
    }else if(tt == trans_type::TT_log){

        return std::unique_ptr<transform<float_t> >(new log_trans<float_t> );
    
    }else{

        throw std::invalid_argument("that transform type was not accounted for");
    
    }
}

////////////////////////////////////////////////////////////////////////////////////

template<typename float_t>
float_t null_trans<float_t>::trans(const float_t& p)
{
    return p;
}


template<typename float_t>
float_t null_trans<float_t>::inv_trans(const float_t& trans_p)
{
    return trans_p;
}


template<typename float_t>
float_t null_trans<float_t>::log_jacobian(const float_t& trans_p)
{
    return 0.0;
}

////////////////////////////////////////////////////////////////////////////////////

template<typename float_t>
float_t twice_fisher_trans<float_t>::trans(const float_t& p)
{
    if ( (p <= -1.0) || (p >= 1.0) )
        throw std::invalid_argument( "error: phi was not between -1 and 1" );
    else
        return std::log(1.0 + p) - std::log(1.0 - p);
}


template<typename float_t>
float_t twice_fisher_trans<float_t>::inv_trans(const float_t &trans_p){
    
    float_t ans = 1.0 - 2.0/(1.0 + std::exp(trans_p));
    if ( (ans <= -1.0) || (ans >= 1.0) )
        throw std::invalid_argument("error: there was probably overflow for exp(trans_p) \n");
    return ans;    
}


template<typename float_t>
float_t twice_fisher_trans<float_t>::log_jacobian(const float_t &trans_p){
    //float_t un_trans_p = inv_trans(trans_p);
    //return std::log(2.0) - std::log(1.0 + un_trans_p) - std::log(1.0 - un_trans_p);
    return std::log(2.0) + trans_p - 2.0*std::log(1.0 + std::exp(trans_p));
}

////////////////////////////////////////////////////////////////////////////////////

template<typename float_t>
float_t logit_trans<float_t>::trans(const float_t& p)
{
    if ( (p < 0.0) || (p > 1.0))
        throw std::invalid_argument("error: p was not between 0 and 1 \n");
    
    return std::log(p) - std::log(1.0 - p);
}


template<typename float_t>
float_t logit_trans<float_t>::inv_trans(const float_t &trans_p){
    
    float_t ans = 1.0/( 1.0 + std::exp(-trans_p) );    
    if ( (ans <= 0.0) || (ans >= 1.0))
        std::cerr << "error: there was probably underflow for exp(-r) \n";
    return ans;
}


template<typename float_t>
float_t logit_trans<float_t>::log_jacobian(const float_t &trans_p){
    return -trans_p - 2.0*std::log(1.0 + std::exp(-trans_p));
}

////////////////////////////////////////////////////////////////////////////////////

template<typename float_t>
float_t log_trans<float_t>::trans(const float_t& p)
{
    if(p < 0.0)
        throw std::invalid_argument("p is negative\n");
    return std::log(p);
}


template<typename float_t>
float_t log_trans<float_t>::inv_trans(const float_t &trans_p){
    return std::exp(trans_p);
}


template<typename float_t>
float_t log_trans<float_t>::log_jacobian(const float_t &trans_p){
    return trans_p;
}

////////////////////////////////////////////////////////////////////////////////////


template<typename float_t>
pack<float_t>::pack(const eig_dyn_vec& trans_params, vecptrs&& t_functors)
    : m_trans_params(trans_params), m_transform_functors(std::move(t_functors))
{
}


template<typename float_t>
pack<float_t>::pack(const eig_dyn_vec &params, const std::vector<trans_type> &vec_trans_types, bool start_w_trans_params)
{
    unsigned int n = vec_trans_types.size(); 
    if( params.rows() != n ){
        throw std::invalid_argument("params and vec_trans_types have to be the same size");
    }
    
    for(auto & tt : vec_trans_types){
        m_transform_functors.push_back(transform<float_t>::create(tt));
    }
    
    if(start_w_trans_params){
        m_trans_params = params;
    }else {
        m_trans_params.resize(n);
        for(size_t i = 0; i < n; ++i)
            m_trans_params(i) = m_transform_functors[i]->trans(params(i));            
    }
}


template<typename float_t>
unsigned int pack<float_t>::get_num_params() const
{
    return m_transform_functors.size();
}


template<typename float_t>
auto pack<float_t>::get_trans_params() const -> eig_dyn_vec 
{
    return m_trans_params.block(0,0,this->get_num_params(),1);
}


template<typename float_t>
auto pack<float_t>::get_untrans_params() const -> eig_dyn_vec
{
    unsigned int n = this->get_num_params();
    eig_dyn_vec params(n);
    for(size_t i = 0; i < n; ++i)
        params(i) = m_transform_functors[i]->inv_trans(m_trans_params(i));
    return params;    
}


template<typename float_t>
auto pack<float_t>::get_trans_params(const unsigned int& start, const unsigned int& end) const -> eig_dyn_vec
{
    return m_trans_params.block(start,0,(end-start+1),1);
}


template<typename float_t>
auto pack<float_t>::get_untrans_params(const unsigned int& start, const unsigned int& end) const -> eig_dyn_vec
{    
    eig_dyn_vec params(end-start+1);
    for(size_t i = start; i < end + 1; ++i)
        params(i-start) = m_transform_functors[i]->inv_trans(m_trans_params(i));
    return params;
}


template<typename float_t>
float_t pack<float_t>::get_log_jacobian() const
{
    float_t result(0.0);
    for(size_t i = 0; i < m_transform_functors.size(); ++i){
        result += m_transform_functors[i]->log_jacobian(m_trans_params(i));
    }
    return result;
}


template<typename float_t>
void pack<float_t>::take_values(const pack<float_t>& other)
{
    if(m_transform_functors.size() != other.get_num_params())
        throw std::invalid_argument("other must have the same size as the caller");
    m_trans_params = other.get_trans_params(0,m_trans_params.size()-1); // TODO: what happens when they are of different sizes?!
}



} //namespace param


#endif // PARAMETERS_H
