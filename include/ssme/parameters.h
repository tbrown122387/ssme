#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <memory> // unique_ptr
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <Eigen/Dense>
#include <vector>
#include <string>
//#include <utility> // std::move
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
     * @brief a clone method that helps with deep-copying transforms
     */
    virtual std::unique_ptr<transform<float_t>> clone() const = 0;


    /**
     * @brief a static method to create unique pointers 
     */
    static std::unique_ptr<transform<float_t> > create(trans_type tt);


    /**
     * @brief a static method to create unique pointers 
     */
    static std::unique_ptr<transform<float_t> > create(const std::string& tt);

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
    std::unique_ptr<transform<float_t>> clone() const override;
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
    std::unique_ptr<transform<float_t>> clone() const override;
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
    std::unique_ptr<transform<float_t>> clone() const override;
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
    std::unique_ptr<transform<float_t>> clone() const override;
};


/**
 * @class pack
 * @author Taylor
 * @brief Stores transformed parameters, as well as the functions 
 * that can change them to the untransformed parameters and back.
 */
template<typename float_t, size_t numelem>
class pack{
private:
 
    // type aliases
    using eig_vec = Eigen::Matrix<float_t,numelem,1>;
    using array_ptrs = std::array<std::unique_ptr<transform<float_t>>, numelem>;
  
    eig_vec m_trans_params;
    array_ptrs m_ts;
    unsigned m_add_idx;

    
public:


    // ctors
    pack(const eig_vec& params, const std::vector<std::string>& transform_names, bool from_transformed = true);


    /**
     * @brief copy ctor
     */
    pack(const pack<float_t,numelem>& other);


    /**
     * @brief default ctor
     */
    pack();


    // assignment operators
    pack<float_t,numelem>& operator=(const pack<float_t,numelem>& other);

    
    void add_param_and_transform(float_t elem, trans_type tt, bool is_transformed = false);    
    
    void add_param_and_transform(float_t elem, const std::string& trans_name, bool is_transformed = false);
   

    /**
     * @brief deep copy a transform
     */
    std::unique_ptr<transform<float_t>> clone_transform(unsigned int idx) const;
   

    /**
     * @brief get the number of elements/transforms
     */
    decltype(auto) size() const;


    /**
     * @brief get the number of elements/transforms
     */
    decltype(auto) capacity() const;    
    

    //! get the transformed parameters in the unrestricted space
    /**
     * @brief get the transformed parameters on the unrestricted space
     * @return an Eigen::Vector of transformed parameters
     */
    auto get_trans_params() const -> eig_vec;


    //! get the untransformed parameters in the possibly-restricted space
    /**
     * @brief get the untransformed parameters on the possibly-restricted space
     * @return an Eigen::Vector of transformed parameters
     */
    auto get_untrans_params() const -> eig_vec;


    //! get a subset of the transformed parameters in the unrestricted space
    // example: pp.get_trans_params(0,2) returns three elements
    /**
     * @brief get a subset of the transformed parameters on the unrestricted space
     * @param index of first element (starts counting at zero)
     * @param index of last element (not like python indexing!)
     * @return an Eigen::Vector of transformed parameters
     */ 
    auto get_trans_params(const unsigned int& start, const unsigned int& end) const -> Eigen::Matrix<float_t, Eigen::Dynamic, 1>;


    //! get a subset of the untransformed parameters in the possibly-restricted space
    // example: pp.get_trans_params(0,2) returns three elements
    /**
    * @brief get a subset of the transformed parameters on the unrestricted space
    * @param index of first element (starts counting at zero)
    * @param index of last element (not like python indexing!)
    * @return an Eigen::Vector of transformed parameters
    */ 
    auto get_untrans_params(const unsigned int& start, const unsigned int& end) const -> Eigen::Matrix<float_t, Eigen::Dynamic, 1>;
    
    
    //! get the log of the Jacobian determinant you need for the density of transformed parameters.
    /**
     * @brief get the log of the Jacobian determinant you need for the density of transformed params
     * @return a float_t
     */
    auto get_log_jacobian() const -> float_t;

    
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

    
template<typename float_t>
std::unique_ptr<transform<float_t> > transform<float_t>::create(const std::string& tt)
{
    if(tt == "null"){
        
        return std::unique_ptr<transform<float_t> >(new null_trans<float_t> );
    
    }else if(tt == "twice_fisher"){
        
        return std::unique_ptr<transform<float_t> >(new twice_fisher_trans<float_t> );
    
    }else if(tt == "logit"){
        
        return std::unique_ptr<transform<float_t> >(new logit_trans<float_t> );
    
    }else if(tt == "log"){

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

    
template<typename float_t>
std::unique_ptr<transform<float_t>> null_trans<float_t>::clone() const
{
    std::unique_ptr<transform<float_t>> r(new null_trans<float_t>(*this));
    return r;
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


template<typename float_t>
std::unique_ptr<transform<float_t>> twice_fisher_trans<float_t>::clone() const
{
    std::unique_ptr<transform<float_t>> r(new twice_fisher_trans<float_t>(*this));
    return r;
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


template<typename float_t>
std::unique_ptr<transform<float_t>> logit_trans<float_t>::clone() const
{
    std::unique_ptr<transform<float_t>> r(new logit_trans<float_t>(*this));
    return r;
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


template<typename float_t>
std::unique_ptr<transform<float_t>> log_trans<float_t>::clone() const
{
    std::unique_ptr<transform<float_t>> r(new log_trans<float_t>(*this));
    return r;
}

////////////////////////////////////////////////////////////////////////////////////


template<typename float_t, size_t numelem>
pack<float_t,numelem>::pack(const eig_vec& params, const std::vector<std::string>& transform_names, bool from_transformed)
{
    // can only create if everything is full length
    if( (params.size() == numelem) && (numelem == transform_names.size())) {
        
        if( from_transformed ){

            m_trans_params = params;
            m_add_idx = numelem;
            for(size_t i = 0; i < numelem; ++i)
                m_ts[i] = transform<float_t>::create(transform_names[i]); 

        }else{
            m_add_idx = numelem;
            for(size_t i = 0; i < numelem; ++i){
                m_ts[i] = transform<float_t>::create(transform_names[i]);
                m_trans_params[i] = m_ts[i]->trans(params[i]); 
            }
        }
    }else{
        throw std::invalid_argument("params needs to be the right size (full)");
    }
}


template<typename float_t, size_t numelem>
pack<float_t,numelem>::pack(const pack<float_t,numelem>& other)
{
    // can only create if everything is full length
    if(other.size() == numelem ) {
        m_add_idx = numelem;
        m_trans_params = other.get_trans_params();
        for(size_t i = 0; i < m_add_idx; ++i)
            m_ts[i] = other.clone_transform(i); 
    }else{
        throw std::invalid_argument("copy ctor can only work with full parameter packs");
    }
}

    
template<typename float_t,size_t numelem>
pack<float_t,numelem>::pack() 
    : m_add_idx(0)
{
}


template<typename float_t,size_t numelem>
void pack<float_t,numelem>::add_param_and_transform(float_t elem, trans_type tt, bool is_transformed)
{
    if( m_add_idx < numelem ){
        m_ts[m_add_idx] = transform<float_t>::create(tt);
        if(is_transformed)
            m_trans_params[m_add_idx] = elem;
        else
            m_trans_params[m_add_idx] = m_ts[m_add_idx]->trans(elem);
        m_add_idx++;
    }else{
        throw std::length_error("can't add any more transformations");
    }
}    


template<typename float_t,size_t numelem>
void pack<float_t,numelem>::add_param_and_transform(float_t elem, const std::string& trans_name, bool is_transformed)
{
    if(m_add_idx < numelem){ 
        m_ts[m_add_idx] = transform<float_t>::create(trans_name);
        if(is_transformed ) m_trans_params[m_add_idx] = elem;
        else m_trans_params[m_add_idx] = m_ts[m_add_idx]->trans(elem);
        m_add_idx++;
    }else{
        throw std::length_error("can't add any more transformations");
    }
}


template<typename float_t, size_t numelem>
std::unique_ptr<transform<float_t>> pack<float_t,numelem>::clone_transform(unsigned int idx) const
{
   return m_ts[idx]->clone(); 
}


template<typename float_t, size_t numelem>
pack<float_t,numelem>& pack<float_t,numelem>::operator=(const pack<float_t,numelem>& other)
{
    // can only create if everything is full length
    if(other.size() == numelem ) {
        m_add_idx = other.size();
        m_trans_params = other.get_trans_params();
        for(size_t i = 0; i < m_add_idx; ++i)
            m_ts[i] = other.clone_transform(i); 
    }else{
        throw std::invalid_argument("pack assignment can only work with full parameter packs");
    }

    return *this;
}


template<typename float_t, size_t numelem>
decltype(auto) pack<float_t,numelem>::size() const 
{
    return m_add_idx;
}


template<typename float_t, size_t numelem>
decltype(auto) pack<float_t,numelem>::capacity() const 
{
    return numelem;
}


template<typename float_t, size_t numelem>
auto pack<float_t,numelem>::get_trans_params() const -> eig_vec 
{
    if(m_add_idx  < numelem)
        throw std::length_error("the parameter container is not full");
    return m_trans_params;
}


template<typename float_t, size_t numelem>
auto pack<float_t,numelem>::get_untrans_params() const -> eig_vec
{
    if(m_add_idx  < numelem) throw std::length_error("the parameter container is not full");
    eig_vec params;
    for(size_t i = 0; i < numelem; ++i)
        params(i) = m_ts[i]->inv_trans(m_trans_params(i));
    return params;    
}


template<typename float_t,size_t numelem>
auto pack<float_t, numelem>::get_trans_params(const unsigned int& start, const unsigned int& end) const -> Eigen::Matrix<float_t, Eigen::Dynamic, 1> 
{
    return m_trans_params.block(start,0,(end-start+1),1);
}


template<typename float_t, size_t numelem>
auto pack<float_t, numelem>::get_untrans_params(const unsigned int& start, const unsigned int& end) const -> Eigen::Matrix<float_t, Eigen::Dynamic, 1> 
{    

    if(m_add_idx < numelem) throw std::length_error("the parameter container is not full");

    Eigen::Matrix<float_t, Eigen::Dynamic, 1> params(end - start + 1);
    size_t i = start;
    do {    
        params(i-start) = m_ts[i]->inv_trans(m_trans_params(i));
        i++;
    } while( i < end);
    return params;
}


template<typename float_t, size_t numelem>
float_t pack<float_t,numelem>::get_log_jacobian() const
{
    if(m_add_idx < numelem) throw std::length_error("the parameter container is not full");

    float_t result(0.0);
    for(size_t i = 0; i < m_add_idx; ++i){
        result += m_ts[i]->log_jacobian(m_trans_params(i));
    }
    return result;
}

  
} //namespace param


#endif // PARAMETERS_H
