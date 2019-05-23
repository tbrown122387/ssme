#ifndef PARAM_TRANSFORMS_H
#define PARAM_TRANSFORMS_H

#include <memory> // unique_ptr
#include <iostream>
#include <stdexcept>
#include <cmath>

/** enum class for different types of transformations */
enum class TransType {TT_null, TT_twice_fisher, TT_logit, TT_log};


/**
 * @class paramTransform
 * @author t
 * @date 22/05/18
 * @file param_transforms.h
 * @brief a pure virtual base class. cts params only.
 */
template<typename float_t>
class paramTransform{
    
public:


    /**
     * @brief a virtual destructor
     */
    virtual ~paramTransform() {}


    /**
     * @brief from the constrained/nontransformed to the transformed/unconstrained space
     */
    virtual float_t trans(const float_t& p) = 0;


    /**
     * @brief from the unconstrained/transformed to the constrained/untransformed space;
     */
    virtual float_t invTrans(const float_t &trans_p) = 0;

    /**
     * @brief get the log jacobian (from untransformed to transformed). For use with adjusting priors to the transformed space.
     */
    virtual float_t logJacobian(const float_t &p) = 0;


    /**
     * @brief a static method to create unique pointers 
     */
    static std::unique_ptr<paramTransform<float_t> > create(TransType tt);
};


/**
 * @class nullTrans
 * @author t
 * @file param_transforms.h
 * @brief trans_p = orig_p.
 */
template<typename float_t>
class nullTrans : public paramTransform<float_t>{
public: 
    float_t trans(const float_t& p) override;
    float_t invTrans(const float_t& trans_p) override;
    float_t logJacobian(const float_t& p) override;
};


/**
 * @class twiceFisherTrans
 * @author t
 * @file param_transforms.h
 * @brief trans_p = log(1+orig_p) - log(1-orig_p) = logit((orig_p+1)/2).
 */
template<typename float_t>
class twiceFisherTrans : public paramTransform<float_t>{
public:
    float_t trans(const float_t& p) override;
    float_t invTrans(const float_t &trans_p) override;   
    float_t logJacobian(const float_t &p) override;
};


/**
 * @class logitTrans
 * @author t
 * @file param_transforms.h
 * @brief trans_p = logit(orig_p).
 */
template<typename float_t>
class logitTrans : public paramTransform<float_t>{
public:
    float_t trans(const float_t& p) override;
    float_t invTrans(const float_t &trans_p) override;
    float_t logJacobian(const float_t &p) override;
};


/**
 * @class logTrans
 * @author t
 * @file param_transforms.h
 * @brief trans_p = log(orig_p).
 */
template<typename float_t>
class logTrans : public paramTransform<float_t>{
public:
    float_t trans(const float_t& p) override;
    float_t invTrans(const float_t &trans_p) override;
    float_t logJacobian(const float_t &p) override;
};


////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////

template<typename float_t>
std::unique_ptr<paramTransform<float_t> > paramTransform<float_t>::create(TransType tt)
{
    if(tt == TransType::TT_null){
        
        return std::unique_ptr<paramTransform<float_t> >(new nullTrans<float_t> );
    
    }else if(tt == TransType::TT_twice_fisher){
        
        return std::unique_ptr<paramTransform<float_t> >(new twiceFisherTrans<float_t> );
    
    }else if(tt == TransType::TT_logit){
        
        return std::unique_ptr<paramTransform<float_t> >(new logitTrans<float_t> );
    
    }else if(tt == TransType::TT_log){

        return std::unique_ptr<paramTransform<float_t> >(new logTrans<float_t> );
    
    }else{

        throw std::invalid_argument("that transform type was not accounted for");
    
    }
}

////////////////////////////////////////////////////////////////////////////////////

template<typename float_t>
float_t nullTrans<float_t>::trans(const float_t& p)
{
    return p;
}


template<typename float_t>
float_t nullTrans<float_t>::invTrans(const float_t& trans_p)
{
    return trans_p;
}


template<typename float_t>
float_t nullTrans<float_t>::logJacobian(const float_t& trans_p)
{
    return 0.0;
}

////////////////////////////////////////////////////////////////////////////////////

template<typename float_t>
float_t twiceFisherTrans<float_t>::trans(const float_t& p)
{
    if ( (p <= -1.0) || (p >= 1.0) )
        throw std::invalid_argument( "error: phi was not between -1 and 1" );
    else
        return std::log(1.0 + p) - std::log(1.0 - p);
}


template<typename float_t>
float_t twiceFisherTrans<float_t>::invTrans(const float_t &trans_p){
    
    float_t ans = 1.0 - 2.0/(1.0 + std::exp(trans_p));
    if ( (ans <= -1.0) || (ans >= 1.0) )
        throw std::invalid_argument("error: there was probably overflow for exp(trans_p) \n");
    return ans;    
}


template<typename float_t>
float_t twiceFisherTrans<float_t>::logJacobian(const float_t &trans_p){
    //float_t un_trans_p = invTrans(trans_p);
    //return std::log(2.0) - std::log(1.0 + un_trans_p) - std::log(1.0 - un_trans_p);
    return std::log(2.0) + trans_p - 2.0*std::log(1.0 + std::exp(trans_p));
}

////////////////////////////////////////////////////////////////////////////////////

template<typename float_t>
float_t logitTrans<float_t>::trans(const float_t& p)
{
    if ( (p < 0.0) || (p > 1.0))
        throw std::invalid_argument("error: p was not between 0 and 1 \n");
    
    return std::log(p) - std::log(1.0 - p);
}


template<typename float_t>
float_t logitTrans<float_t>::invTrans(const float_t &trans_p){
    
    float_t ans = 1.0/( 1.0 + std::exp(-trans_p) );    
    if ( (ans <= 0.0) || (ans >= 1.0))
        std::cerr << "error: there was probably underflow for exp(-r) \n";
    return ans;
}


template<typename float_t>
float_t logitTrans<float_t>::logJacobian(const float_t &trans_p){
    //float_t un_trans_p = invTrans(trans_p);
    //return -std::log(1.0-un_trans_p) - std::log(un_trans_p);
    return -trans_p - 2.0*std::log(1.0 + std::exp(-trans_p));
}

////////////////////////////////////////////////////////////////////////////////////

template<typename float_t>
float_t logTrans<float_t>::trans(const float_t& p)
{
    if(p < 0.0)
        throw std::invalid_argument("p is negative\n");
    return std::log(p);
}


template<typename float_t>
float_t logTrans<float_t>::invTrans(const float_t &trans_p){
    return std::exp(trans_p);
}


template<typename float_t>
float_t logTrans<float_t>::logJacobian(const float_t &trans_p){
    //float_t un_trans_p = invTrans(trans_p);
    //return -std::log(un_trans_p);
    return trans_p;
}

////////////////////////////////////////////////////////////////////////////////////


#endif //PARAM_TRANSFORMS_H
