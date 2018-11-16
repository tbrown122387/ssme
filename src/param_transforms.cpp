#include "param_transforms.h"

#include <iostream>
#include <stdexcept>
#include <cmath>


////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<paramTransform> paramTransform::create(TransType tt)
{
    if(tt == TransType::TT_null){
        
        return std::unique_ptr<paramTransform>(new nullTrans);
    
    }else if(tt == TransType::TT_twice_fisher){
        
        return std::unique_ptr<paramTransform>(new twiceFisherTrans);
    
    }else if(tt == TransType::TT_logit){
        
        return std::unique_ptr<paramTransform>(new logitTrans);
    
    }else if(tt == TransType::TT_log){

        return std::unique_ptr<paramTransform>(new logTrans);
    
    }else{

        throw std::invalid_argument("that transform type was not accounted for");
    
    }
}

////////////////////////////////////////////////////////////////////////////////////

double nullTrans::trans(const double& p)
{
    return p;
}


double nullTrans::invTrans(const double& trans_p)
{
    return trans_p;
}


double nullTrans::logJacobian(const double& trans_p)
{
    return 0.0;
}

////////////////////////////////////////////////////////////////////////////////////

double twiceFisherTrans::trans(const double& p)
{
    if ( (p <= -1.0) || (p >= 1.0) )
        throw std::invalid_argument( "error: phi was not between -1 and 1" );
    else
        return std::log(1.0 + p) - std::log(1.0 - p);
}


double twiceFisherTrans::invTrans(const double &trans_p){
    
    double ans = 1.0 - 2.0/(1.0 + std::exp(trans_p));
    if ( (ans <= -1.0) || (ans >= 1.0) )
        throw std::invalid_argument("error: there was probably overflow for exp(trans_p) \n");
    return ans;    
}


double twiceFisherTrans::logJacobian(const double &trans_p){
    //double un_trans_p = invTrans(trans_p);
    //return std::log(2.0) - std::log(1.0 + un_trans_p) - std::log(1.0 - un_trans_p);
    return std::log(2.0) + trans_p - 2.0*std::log(1.0 + std::exp(trans_p));
}

////////////////////////////////////////////////////////////////////////////////////

double logitTrans::trans(const double& p)
{
    if ( (p < 0.0) || (p > 1.0))
        throw std::invalid_argument("error: p was not between 0 and 1 \n");
    
    return std::log(p) - std::log(1.0 - p);
}


double logitTrans::invTrans(const double &trans_p){
    
    double ans = 1.0/( 1.0 + std::exp(-trans_p) );    
    if ( (ans <= 0.0) || (ans >= 1.0))
        std::cerr << "error: there was probably underflow for exp(-r) \n";
    return ans;
}


double logitTrans::logJacobian(const double &trans_p){
    //double un_trans_p = invTrans(trans_p);
    //return -std::log(1.0-un_trans_p) - std::log(un_trans_p);
    return -trans_p - 2.0*std::log(1.0 + std::exp(-trans_p));
}

////////////////////////////////////////////////////////////////////////////////////

double logTrans::trans(const double& p)
{
    if(p < 0.0)
        throw std::invalid_argument("p is negative\n");
    return std::log(p);
}


double logTrans::invTrans(const double &trans_p){
    return std::exp(trans_p);
}


double logTrans::logJacobian(const double &trans_p){
    //double un_trans_p = invTrans(trans_p);
    //return -std::log(un_trans_p);
    return trans_p;
}

////////////////////////////////////////////////////////////////////////////////////
