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
 * @class transform_container
 * @brief holds a bunch of (pointers to) transform function objects
 */
template<typename float_t, size_t numelem>
class transform_container{
private:

    using array_ptrs = std::array<std::unique_ptr<transform<float_t>>, numelem>;
    array_ptrs m_ts;
    unsigned m_add_idx;

public:
 
    /**
     * @brief ctor
     */
    transform_container();
  

    /**
     * @brief copy ctor
     * @param ts the transform container you want to copy
     */
    transform_container(const transform_container<float_t,numelem>& ts);
   

    /**
     * @brief ctor
     */
    transform_container(std::vector<std::string> trans_names);


    /**
     * @brief assignment operator
     */
    transform_container<float_t,numelem>& operator=(const transform_container<float_t,numelem>& other);
   

    /**
     * @brief add a transform to the container
     */
    void add_transform(trans_type tt);


    /**
     * @brief get the transforms as an array of pointers
     */
    auto get_transforms() const -> array_ptrs;


    /**
     * @brief get the number of elements/transforms
     */
    decltype(auto) size() const;


    /**
     * @brief get the number of elements/transforms
     */
    decltype(auto) capacity() const;


    /**
     * @brief access a specific pointer to a transform
     */
    std::unique_ptr<transform<float_t>> operator[](unsigned int i) const; 
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
   
    eig_vec m_trans_params;
    transform_container<float_t, numelem> m_transform_functors;
    
public:

    // ctors
    pack(const eig_vec &trans_params, const transform_container<float_t, numelem>& t_functors, bool from_transformed = true);


    //TODO: define brace initialization
    /**
     * @brief ctor
     * @param params an Eigen vector of parameters (transformed or nontransformed)
     * @param vec_trans_types a vector of smooth transformations
     * @param start_w_trans_params whether or not you are instantiating with (non)transformed params
     */
//    pack(const eig_dyn_vec &params, 
//         const std::vector<trans_type> &vec_trans_types, 
//         bool start_w_trans_params = true);
   
    
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

    
    //! get number of parameters
    /**
     * @brief gets the number of parameters in your pack
     * @return an unsigned integer 
     */
    decltype(auto) get_num_params() const;


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


    /**
     * @brief getter for the smooth transforms
     */
    auto get_transforms() const -> transform_container<float_t,numelem>;


    
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
transform_container<float_t,numelem>::transform_container()
    : m_add_idx(0)
{
}


template<typename float_t, size_t numelem>
transform_container<float_t,numelem>::transform_container(const transform_container<float_t,numelem>& ts)
{
    m_ts = ts.get_transforms();
    m_add_idx = ts.size();
}


template<typename float_t, size_t numelem>
transform_container<float_t,numelem>::transform_container(std::vector<std::string> trans_names)
{
    m_add_idx = 0;
    for(auto& name : trans_names){
        m_ts[m_add_idx] = transform<float_t>::create(name);
        m_add_idx++; 
    }
}


template<typename float_t, size_t numelem>
transform_container<float_t,numelem>& transform_container<float_t,numelem>::operator=(const transform_container<float_t,numelem>& other)
{
    if( this != &other){
        m_ts = other.get_transforms();
        m_add_idx = other.size();
    }
    return *this; 
}


template<typename float_t, size_t numelem>
void transform_container<float_t, numelem>::add_transform(trans_type tt)
{
    m_ts[m_add_idx] = transform<float_t>::create(tt);
    m_add_idx++;
}


template<typename float_t, size_t numelem>
auto transform_container<float_t,numelem>::get_transforms() const -> array_ptrs
{
    array_ptrs deep_cpy;
    for(size_t i = 0; i < m_add_idx; ++i)
        deep_cpy[i] = m_ts[i]->clone();
    
    return deep_cpy;
}


template<typename float_t, size_t numelem>
decltype(auto) transform_container<float_t,numelem>::size() const 
{
   return m_add_idx; 
}


template<typename float_t, size_t numelem>
decltype(auto) transform_container<float_t,numelem>::capacity() const 
{
   return numelem; 
}


template<typename float_t, size_t numelem>
std::unique_ptr<transform<float_t>> transform_container<float_t, numelem>::operator[](unsigned int i) const
{
   return m_ts[i]->clone(); 
} 

////////////////////////////////////////////////////////////////////////////////////


template<typename float_t, size_t numelem>
pack<float_t,numelem>::pack(const eig_vec& params, const transform_container<float_t,numelem>& t_functors, bool from_transformed)
    : m_transform_functors(t_functors)
{
    if(from_transformed)
        m_trans_params = params;
    else{
        auto ts = this->get_transforms();
        for(size_t i = 0; i < numelem; ++i){
            m_trans_params[i] = ts[i]->trans(params[i]);
        }
    }
}


///template<typename float_t>
///pack<float_t>::pack(const eig_dyn_vec &params, 
///                    const std::vector<trans_type> &vec_trans_types, 
///                    bool start_w_trans_params)
///{
///    auto n = vec_trans_types.size(); 
///    if( params.rows() != n ){
///        throw std::invalid_argument("params and vec_trans_types have to be the same size");
///    }
///    
///    for(auto & tt : vec_trans_types)
///        m_transform_functors.add_transform(tt);
///    
///    if(start_w_trans_params){
///        m_trans_params = params;
///    }else {
///        m_trans_params.resize(n);
///        for(size_t i = 0; i < n; ++i)
///            m_trans_params(i) = m_transform_functors[i]->trans(params(i));            
///    }
///}


template<typename float_t, size_t numelem>
pack<float_t,numelem>::pack(const pack<float_t,numelem>& other)
{
    m_trans_params = other.get_trans_params();
    m_transform_functors = other.get_transforms();
}

    
template<typename float_t,size_t numelem>
pack<float_t,numelem>::pack()
{
}


template<typename float_t, size_t numelem>
pack<float_t,numelem>& pack<float_t,numelem>::operator=(const pack<float_t,numelem>& other)
{
    if( this != &other ){
        m_trans_params = other.get_trans_params();
        m_transform_functors = other.get_transforms();
    }
    return *this;
}


template<typename float_t, size_t numelem>
decltype(auto) pack<float_t,numelem>::get_num_params() const 
{
    return m_transform_functors.size();
}


template<typename float_t, size_t numelem>
auto pack<float_t,numelem>::get_trans_params() const -> eig_vec 
{
    return m_trans_params.block(0,0,this->get_num_params(),1);
}


template<typename float_t, size_t numelem>
auto pack<float_t,numelem>::get_untrans_params() const -> eig_vec
{
    eig_vec params;
    for(size_t i = 0; i < numelem; ++i)
        params(i) = m_transform_functors[i]->inv_trans(m_trans_params(i));
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
    Eigen::Matrix<float_t, Eigen::Dynamic, 1> params(end - start + 1);
    //for(size_t i = start; i < end + 1; ++i)
    size_t i = start;
    do {    
        params(i-start) = m_transform_functors[i]->inv_trans(m_trans_params(i));
        i++;
    } while( i < end);
    return params;
}


template<typename float_t, size_t numelem>
float_t pack<float_t,numelem>::get_log_jacobian() const
{
    float_t result(0.0);
    for(size_t i = 0; i < m_transform_functors.size(); ++i){
        result += m_transform_functors[i]->log_jacobian(m_trans_params(i));
    }
    return result;
}

  
template<typename float_t, size_t numelem>
auto pack<float_t, numelem>::get_transforms() const -> transform_container<float_t,numelem>
{
    return transform_container<float_t,numelem>{ m_transform_functors };
}


} //namespace param


#endif // PARAMETERS_H
