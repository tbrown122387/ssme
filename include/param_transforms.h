#ifndef PARAM_TRANSFORMS_H
#define PARAM_TRANSFORMS_H

#include <memory> // unique_ptr

/** enum class for different types of transformations */
enum class TransType {TT_null, TT_twice_fisher, TT_logit, TT_log};


/**
 * @class paramInvTransform
 * @author t
 * @date 22/05/18
 * @file param_pack.h
 * @brief invTrans() gives you the contrained/untransformed version, 
 * and logJacobian() returns the logJacobian of the forward/non-inv. trans(to the unconstrained).
 */
class paramInvTransform{
    
public:
    virtual ~paramInvTransform() {}
    virtual double trans(const double& p) = 0;
    virtual double invTrans(const double &trans_p) = 0;
    virtual double logJacobian(const double &trans_p) = 0;
    static std::unique_ptr<paramInvTransform> create(TransType tt);
};


class nullTrans : public paramInvTransform{
public: 
    double trans(const double& p) override;
    double invTrans(const double& trans_p) override;
    double logJacobian(const double& trans_p) override;
};


class twiceFisherTrans : public paramInvTransform{
public:
    double trans(const double& p) override;
    double invTrans(const double &trans_p) override;   
    double logJacobian(const double &trans_p) override;
};


class logitTrans : public paramInvTransform{
public:
    double trans(const double& p) override;
    double invTrans(const double &trans_p) override;
    double logJacobian(const double &trans_p) override;
};


class logTrans : public paramInvTransform{
public:
    double trans(const double& p) override;
    double invTrans(const double &trans_p) override;
    double logJacobian(const double &trans_p) override;
};


#endif //PARAM_TRANSFORMS_H