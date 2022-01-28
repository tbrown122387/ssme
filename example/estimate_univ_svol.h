#ifndef ESTIMATE_UNIV_SVOL_H
#define ESTIMATE_UNIV_SVOL_H

#include <string>

#include <ssme/ada_pmmh_mvn.h>
#include <ssme/parameters.h>
#include <pf/rv_eval.h>
#include <pf/resamplers.h>

#include "univ_svol_bootstrap_filter.h"


using namespace pf::resamplers;


template<size_t numparams, size_t dimstate, size_t dimobs, size_t numparts, typename float_t>
class univ_svol_estimator : public ada_pmmh_mvn<numparams,dimobs,numparts,float_t>
{
public:

    using psv = Eigen::Matrix<float_t,numparams,1>;
    using psm = Eigen::Matrix<float_t,numparams,numparams>;
    using osv = Eigen::Matrix<float_t,dimobs,1>;
    
    univ_svol_estimator(
            const psv &start_trans_theta,
            std::vector<std::string> tts,
            const unsigned int &num_mcmc_iters,
            const unsigned int &num_pfilters, 
            const std::string &data_file, 
            const std::string &samples_base_name, 
            const std::string &messages_base_name,
            const bool &mc,
            const unsigned int &t0, 
            const unsigned int &t1,
            const psm &C0,
            bool print_to_console,
            unsigned int print_every_k);
        
    float_t log_prior_eval(const param::pack<float_t,3>& theta);

    float_t log_like_eval(const param::pack<float_t,3>& theta, const std::vector<osv> &data);

};

template<size_t numparams, size_t dimstate, size_t dimobs, size_t numparts, typename float_t>
univ_svol_estimator<numparams,dimstate,dimobs,numparts,float_t>::univ_svol_estimator(
                                                    const psv &start_trans_theta,
                                                    std::vector<std::string> tts,
                                                    const unsigned int &num_mcmc_iters,
                                                    const unsigned int &num_pfilters, 
                                                    const std::string &data_file, 
                                                    const std::string &samples_base_name, 
                                                    const std::string &messages_base_name,
                                                    const bool &mc,
                                                    const unsigned int &t0,
                                                    const unsigned int &t1,
                                                    const psm &C0,
                                                    bool print_to_console,
                                                    unsigned int print_every_k) 
    : ada_pmmh_mvn<numparams,dimobs,numparts,float_t>(start_trans_theta, 
                                                      tts, 
                                                      num_mcmc_iters, 
                                                      num_pfilters,
                                                      data_file, 
                                                      samples_base_name, 
                                                      messages_base_name, 
                                                      mc, 
                                                      t0, 
                                                      t1, 
                                                      C0, 
                                                      print_to_console,
                                                      print_every_k)
{
}


template<size_t numparams, size_t dimstate, size_t dimobs, size_t numparts, typename float_t>
float_t univ_svol_estimator<numparams,dimstate,dimobs,numparts,float_t>::log_prior_eval(const param::pack<float_t,3>& theta)
{
    // value to be returned
    float_t returnThis(0.0);
   
    // 1 beta, 1 phi, 1 ss
    // unpack parameters
    float_t beta = theta.get_untrans_params(0,0)(0);
    float_t phi  = theta.get_untrans_params(1,1)(0);
    float_t ss   = theta.get_untrans_params(2,2)(0);
    
    // beta ~ normal(1.0, 1.0)
    returnThis += rveval::evalUnivNorm<float_t>(beta, 1.0, 1.0, true);

    // phi ~ Uniform(0,1)
    returnThis += rveval::evalUniform<float_t>(phi, 0.0, 1.0, true);

    // ss ~ InverseGamma(.001, .001)
    returnThis += rveval::evalUnivInvGamma<float_t>(ss, .001, .001, true);

    return returnThis;    
}


template<size_t numparams, size_t dimstate, size_t dimobs, size_t numparts, typename float_t>
float_t univ_svol_estimator<numparams,dimstate,dimobs,numparts,float_t>::log_like_eval(const param::pack<float_t,3>& theta, const std::vector<osv> &data)
{

    // jump out if there's a problem with the data
    if(data.empty())
        throw std::length_error("can't read in data\n");

    // the value to be returned
    float_t logLike(0.0);
    
    // instantiate model
    svol_bs<numparts, dimstate, dimobs, mn_resampler<numparts,dimstate,float_t>,float_t> mod(theta);
    
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


template<size_t numparams, size_t dimstate, size_t dimobs, size_t numparts, typename float_t>
void do_ada_pmmh_univ_svol(const std::string &datafile, 
                           const std::string &samples_base_name, 
                           const std::string &messages_base_name, 
                           unsigned int num_mcmc_iters, 
                           unsigned int num_pfilters,
                           bool multicore)
{

    // the chain's starting parameters
    using psv = Eigen::Matrix<float_t,numparams,1>;
    using psm = Eigen::Matrix<float_t,numparams,numparams>;

    psv start_trans_theta;
    start_trans_theta << 1.0, rveval::twiceFisher<float_t>(.5), std::log(2.0e-4);

    std::vector<std::string> tts {"null", "twice_fisher", "log"}; // betas phis sigma squareds 
    
    // the chain's initial covariance matrix 
    psm C0 = psm::Identity()*.15;
    unsigned int t0 = 150;  // start adapting the covariance at this iteration
    unsigned int t1 = 1000; // end adapting the covariance at this iteration
    univ_svol_estimator<numparams,dimstate,dimobs,numparts,float_t> mcmcobj(
                                                    	start_trans_theta,
                                                    	tts,
                                                    	num_mcmc_iters, 
                                                        num_pfilters,
                                                    	datafile, 
                                                    	samples_base_name, 
                                                    	messages_base_name, 
                                                    	multicore, 
                                                    	t0,
                                                    	t1,
                                                    	C0,
                                                    	false, // print console
                                                        1);
    mcmcobj.commence_sampling();

}



#endif //ESTIMATE_MSL1_H
