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
 * @brief a pure virtual base class. cts params only.
 */
class paramTransform{
    
public:


    /**
     * @brief a virtual destructor
     */
    virtual ~paramTransform() {}


    /**
     * @brief from the constrained/nontransformed to the transformed/unconstrained space
     */
    virtual double trans(const double& p) = 0;


    /**
     * @brief from the unconstrained/transformed to the constrained/untransformed space;
     */
    virtual double invTrans(const double &trans_p) = 0;

    /**
     * @brief get the log jacobian (from untransformed to transformed). For use with adjusting priors to the transformed space.
     */
    virtual double logJacobian(const double &p) = 0;


    /**
     * @brief a static method to create unique pointers 
     */
    static std::unique_ptr<paramTransform> create(TransType tt);
};


/**
 * @class paramInvTransform
 * @author t
 * @file param_pack.h
 * @brief trans_p = orig_p.
 */
class nullTrans : public paramTransform{
public: 
    double trans(const double& p) override;
    double invTrans(const double& trans_p) override;
    double logJacobian(const double& p) override;
};


/**
 * @class paramInvTransform
 * @author t
 * @file param_pack.h
 * @brief trans_p = log(1+orig_p) - log(1-orig_p) = logit((orig_p+1)/2).
 */
class twiceFisherTrans : public paramTransform{
public:
    double trans(const double& p) override;
    double invTrans(const double &trans_p) override;   
    double logJacobian(const double &p) override;
};


/**
 * @class paramInvTransform
 * @author t
 * @file param_pack.h
 * @brief trans_p = logit(orig_p).
 */
class logitTrans : public paramTransform{
public:
    double trans(const double& p) override;
    double invTrans(const double &trans_p) override;
    double logJacobian(const double &p) override;
};


/**
 * @class paramInvTransform
 * @author t
 * @file param_pack.h
 * @brief trans_p = log(orig_p).
 */
class logTrans : public paramTransform{
public:
    double trans(const double& p) override;
    double invTrans(const double &trans_p) override;
    double logJacobian(const double &p) override;
};


#endif //PARAM_TRANSFORMS_H
