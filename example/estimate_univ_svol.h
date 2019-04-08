#ifndef ESTIMATE_UNIV_SVOL_H
#define ESTIMATE_UNIV_SVOL_H

#include <string>

#include "ada_pmmh_mvn.h"
#include "param_pack.h"
#include "rv_eval.h"
#include "resamplers.h"
#include "univ_svol_bootstrap_filter.h"


//////////////////////////////////////////////////////////////////////////////////////////
//////////////////// univ_svol_estimator /////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

template<size_t numparams, size_t dimstate, size_t dimobs, size_t numparts>
class univ_svol_estimator : public ada_pmmh_mvn<numparams,dimobs,numparts>
{
public:

    using psv = Eigen::Matrix<double,numparams,1>;
    using psm = Eigen::Matrix<double,numparams,numparams>;
    using osv = Eigen::Matrix<double,dimobs,1>;
    
    univ_svol_estimator(
            const psv &startTransTheta,
            const std::vector<TransType>& tts,
            const unsigned int &numMCMCIters, 
            const std::string &dataFile, 
            const std::string &samples_base_name, 
            const std::string &messages_base_name,
            const bool &mc,
            const unsigned int &t0, 
            const unsigned int &t1,
            const psm &C0,
            bool print_to_console);
        
    double logPriorEvaluate(const paramPack& theta);

    double logLikeEvaluate(const paramPack& theta, const std::vector<osv> &data);

};

template<size_t numparams, size_t dimstate, size_t dimobs, size_t numparts>
univ_svol_estimator<numparams,dimstate,dimobs,numparts>::univ_svol_estimator(
                                                    const psv &startTransTheta,
                                                    const std::vector<TransType>& tts,
                                                    const unsigned int &numMCMCIters, 
                                                    const std::string &dataFile, 
                                                    const std::string &samples_base_name, 
                                                    const std::string &messages_base_name,
                                                    const bool &mc,
                                                    const unsigned int &t0,
                                                    const unsigned int &t1,
                                                    const psm &C0,
                                                    bool print_to_console) 
    : ada_pmmh_mvn<numparams,dimobs,numparts>(startTransTheta, tts, numMCMCIters, dataFile, samples_base_name, messages_base_name, mc, t0, t1, C0, print_to_console)
{
}


template<size_t numparams, size_t dimstate, size_t dimobs, size_t numparts>
double univ_svol_estimator<numparams,dimstate,dimobs,numparts>::logPriorEvaluate(const paramPack& theta)
{
    // value to be returned
    double returnThis(0.0);
   
    // 1 beta, 1 phi, 1 ss
    // unpack parameters
    double beta = theta.getUnTransParams(0,0)(0);
    double phi  = theta.getUnTransParams(1,1)(0);
    double ss   = theta.getUnTransParams(2,2)(0);
    
    // beta ~ normal(1.0, 1.0)
    returnThis += rveval::evalUnivNorm(beta, 1.0, 1.0, true);

    // phi ~ Uniform(0,1)
    returnThis += rveval::evalUniform(phi, 0.0, 1.0, true);

    // ss ~ InverseGamma(.001, .001)
    returnThis += rveval::evalUnivInvGamma(ss, .001, .001, true);

    return returnThis;    
}


template<size_t numparams, size_t dimstate, size_t dimobs, size_t numparts>
double univ_svol_estimator<numparams,dimstate,dimobs,numparts>::logLikeEvaluate(const paramPack& theta, const std::vector<osv> &data)
{

    // jump out if there's a problem with the data
    if(data.empty())
        throw std::length_error("can't read in data\n");

    // the value to be returned
    double logLike(0.0);
    
    // instantiate model
    svol_bs<numparts, dimstate, dimobs, mn_resampler<numparts,dimstate>> mod(theta);
    
    // iterate through data and calculate sum of log p(y_t | y_{1:t-1}) s
    unsigned int row (0);
    while(row < data.size()){
        mod.filter(data[row]);  
        logLike += mod.getLogCondLike();
        row++;
    }

    // return the estimate of the log likelihood
    return logLike;    
}


//////////////////////////////////////////////////////////////////////////////////////////
//////////////////// do_ada_pmmh_msl1 ////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////


template<size_t numparams, size_t dimstate, size_t dimobs, size_t numparts>
void do_ada_pmmh_univ_svol(const std::string &datafile, const std::string &samples_base_name, const std::string &messages_base_name, unsigned int num_mcmc_iters, bool multicore)
{
    // the chain's starting parameters
    using psv = Eigen::Matrix<double,numparams,1>;
    using psm = Eigen::Matrix<double,numparams,numparams>;
    psv start_trans_theta;
    start_trans_theta << 1.0, rveval::twiceFisher(.5), std::log(2.0e-4);
    std::vector<TransType> tts;
    tts.push_back(TransType::TT_null); // betas
    tts.push_back(TransType::TT_twice_fisher); // phis
    tts.push_back(TransType::TT_log); // sigma squareds
   
    // the chain's initial covariance matrix 
    psm C0 = psm::Identity()*.15;
    unsigned int t0 = 150;  // start adapting the covariance at this iteration
    unsigned int t1 = 1000; // end adapting the covariance at this iteration
    univ_svol_estimator<numparams,dimstate,dimobs,numparts> mcmcobj(
                                                    	start_trans_theta,
                                                    	tts,
                                                    	num_mcmc_iters, 
                                                    	datafile, 
                                                    	samples_base_name, 
                                                    	messages_base_name, 
                                                    	multicore, 
                                                    	t0,
                                                    	t1,
                                                    	C0,
                                                    	true );
    mcmcobj.commenceSampling();

}



#endif //ESTIMATE_MSL1_H
