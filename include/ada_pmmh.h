#ifndef ADA_PMMH_H
#define ADA_PMMH_H


#include <vector>
#include <Eigen/Dense>
#include <iostream> // ofstream
#include <thread> // hardware_concurrency
#include <atomic> // atomic_bool
#include <fstream> // ofstream
#include <future> // std::future

#include "rv_samp.h"
#include "utils.h" // readInData


/**
 * @class ada_pmmh
 * @author t
 * @file ada_pmmh.h
 * @brief Performs an adaptive version of the particle marginal Metropilis-Hastings
 * algorithm. The user is asked to design his/her own proposal density and prior 
 * distribution. The sampling is handled on an "as-is" basis, which means that it is
 * entirely the user's own responsibility to handle Jacobians that might pop up if 
 * the prior and proposal density are specified for two different parameterizations.
 * These parameters will be written in an "as-is" fashion, as well. These calculations 
 * can be facilitates using the `paramPack` class, however using this class is not 
 * absolutely necessary. Also, for convenience, a moving average covariance matrix 
 * estimate is performed. This can be ignored, but it can be handy for use in the
 * proposal density on a transformed space.
 */
template<size_t numparams, size_t dimobs, size_t numparts, typename float_t>
class ada_pmmh{
public:
// TODO: optional logging!

    using osv = Eigen::Matrix<float_t,dimobs,1>;
    using psv = Eigen::Matrix<float_t,numparams,1>;
    using psm = Eigen::Matrix<float_t,numparams,numparams>;
    

    /**
     * @brief The constructor.
     * @param start_trans_theta the initial (possibly transformed) parameters 
     * you want to start sampling from.
     * @param num_mcmc_iters the number of MCMC iterations you want to do.
     * @param data_file the location of the observed time series data.
     * @param samples_file the location where you want to store the samples.
     * @param messages_file the location where you want to store the messages.
     * @param mc stands for multicore. true or false if you want to use extra cores.
     * @param t0 iteration at which you start adapting the posterior covariance matrix.
     * @param t1 time you stop adapting the posterior covariance matrix.
     * @param C0 initial covariance matrix to be used in the moving ave. calculation.
     */
    ada_pmmh(const psv &start_trans_theta, 
             const unsigned int &num_mcmc_iters,
             const std::string &data_file, 
             const std::string &samples_file, 
             const std::string &messages_file,
             const bool &mc,
             const unsigned int &t0,
             const unsigned int &t1,
             const psm &C0);
             

    /**
     * @brief Get the current proposal distribution's covariance matrix.
     * @return the covariance matrix of q(theta' | theta)
     */
    psm get_ct() const;
    

    /**
     * @brief starts the sampling
     */
    void commenceSampling();
    

    /**
     * @brief Evaluates the log of the model's prior distribution.
     * @param transTheta the transformed parameters argument.
     * @return the log of the prior density.
     */
    virtual float_t logPriorEvaluate(const psv &transTheta) = 0;


    /** 
     * @brief Evaluates the logarithm of the proposal density.
     * @param oldTransParams the old parameters that are (probably) transformed.
     * @param newTransParams the new parameters that are (probably) transformed. 
     * @return the log of the proposal density.
     */
    virtual float_t logQEvaluate(const psv &oldTransParams, const psv &newTransParams) = 0; 


    /**
     * @brief Approximates the log likelihood with a particle filter.
     * @param theta the parameters with which to run the particle filter.
     * @param data the observed data with which to run the particle filter.
     * @param cancelled is a token you need to provide if doing multithreaded likelihood evals. This allows the function to terminate prematurely. 
     * @return the evaluation of the log likelihood approximation.
     */
    virtual float_t logLikeEvaluate(const psv &transTheta, const std::vector<osv> &data, std::atomic_bool& cancelled) = 0;
//TODO: remove cancellation token
    
             
private:
    std::vector<osv> m_data;
    psv m_current_trans_theta;
    psm m_sigma_hat; // for transformed parameters; n-1 in the denominator.
    psv m_mean_trans_theta;
    float_t m_ma_accept_rate;
    unsigned int m_t0;  // the time it starts adapting
    unsigned int m_t1; // the time it stops adapting
    psm m_Ct;
    rvsamp::MVNSampler<numparams> m_mvn_gen;
    std::ofstream m_samples_file_stream; 
    std::ofstream m_message_stream;
    unsigned int m_num_mcmc_iters;
    unsigned int m_num_extra_threads;
//    std::mutex m_outFileMutex;
    bool m_multicore;
    unsigned int m_iter; // current iter
    float_t m_sd; // perhaps there's a better name for this
    float_t m_eps;
    
    
    void update_moments_and_Ct(const psv &newTransTheta);
    psv qSample(const psv &oldTransTheta);
};


template<size_t numparams, size_t dimobs, size_t numparts, typename float_t>
ada_pmmh<numparams,dimobs,numparts,float_t>::ada_pmmh(const psv &start_trans_theta, 
                                              const unsigned int &num_mcmc_iters,
                                              const std::string &data_file, 
                                              const std::string &samples_file, 
                                              const std::string &messages_file,
                                              const bool &mc,
                                              const unsigned int &t0,
                                              const unsigned int &t1,
                                              const psm &C0)
 : m_current_trans_theta(start_trans_theta)
 , m_sigma_hat(psm::Zero())
 , m_mean_trans_theta(psv::Zero())
 , m_ma_accept_rate(0.0)
 , m_t0(t0), m_t1(t1)
 , m_Ct(C0)
 , m_num_mcmc_iters(num_mcmc_iters)
 , m_multicore(mc)
 , m_iter(0)
 , m_sd(2.4*2.4/numparams)
 , m_eps(.01)
{
    m_data = utils::readInData<dimobs,float_t>(data_file);
    m_samples_file_stream.open(samples_file);
    m_message_stream.open(messages_file);  
    m_num_extra_threads = std::thread::hardware_concurrency() - 1;
}


template<size_t numparams, size_t dimobs, size_t numparts, typename float_t>
void ada_pmmh<numparams,dimobs,numparts,float_t>::update_moments_and_Ct(const psv &newTransTheta)  
{
    // if m_iter = 1, that means we're on iteration 2, 
    // but we're calling this based on the previous iteration, 
    // so that's iteration 1, think of n as the actual 
    // number of iterations that have happened then (not counting from 0)
    // On the other hand, yu might not want to count the first iteration, 
    // though, because we didn't "propose" anything and probabilistically accept/reject it
    if(m_iter == 1){
        // at the moment m_meanTransTheta is zero, so we add m_currentTransTheta
        // we'll also leave m_sigma_hat to be zero for the time being
        m_mean_trans_theta += newTransTheta;
    }else if(m_iter == 2){
        // at the beginning of this call, m_mean_trans_theta is x1*x1^T and m_sigma_hat is zero
        // this is a an algebraically simplified version of the split up non-recursive formula
        m_sigma_hat = m_mean_trans_theta*m_mean_trans_theta.transpose() 
                    + newTransTheta * newTransTheta.transpose() 
                    - m_mean_trans_theta * newTransTheta.transpose() 
                    - newTransTheta * m_mean_trans_theta.transpose();
        m_sigma_hat *= .5;
        m_mean_trans_theta = .5* m_mean_trans_theta + .5*newTransTheta;     
    }else if( m_iter > 2){
        // regular formula. see
        // https://stats.stackexchange.com/questions/310680/sequential-recursive-online-calculation-of-sample-covariance-matrix/310701#310701
        m_sigma_hat = m_sigma_hat*(m_iter-2.0)/(m_iter-1.0) 
                    + (newTransTheta - m_mean_trans_theta)*(newTransTheta - m_mean_trans_theta).transpose()/m_iter;
        m_mean_trans_theta = ((m_iter-1.0)*m_mean_trans_theta + newTransTheta)/m_iter;

    }else{
        std::cerr << "something went wrong\n";
    }
    
    // now update Ct
    if( (m_t1 > m_iter) && (m_iter > m_t0) ) // in window, so we want to adjust Ct
        m_Ct = m_sd * ( m_sigma_hat + m_eps * psm::Identity() );

}


template<size_t numparams, size_t dimobs, size_t numparts, typename float_t>
auto ada_pmmh<numparams,dimobs,numparts,float_t>::get_ct() const -> psm
{
    return m_Ct;
}


template<size_t numparams, size_t dimobs, size_t numparts>
auto ada_pmmh<numparams,dimobs,numparts,float_t>::qSample(const psv &oldTransParams) -> psv
{
    // assumes that Ct has already been updated
    // assumes parameters are in transformed space    
    m_mvn_gen.setMean(oldTransParams);
    m_mvn_gen.setCovar(m_Ct);
    return m_mvn_gen.sample();
}


template<size_t numparams, size_t dimobs, size_t numparts, typename float_t>
void ada_pmmh<numparams,dimobs,numparts,float_t>::commenceSampling()
{

    // random number stuff to decide on whether to accept or reject
    rvsamp::UniformSampler runif; 
    
    float_t oldLogLike (0.0);
    float_t oldLogPrior(0.0);
    while(m_iter < m_num_mcmc_iters) // every iteration
    {        

        // first iteration no acceptance probability
        if (m_iter == 0) { 
            
            m_message_stream << "***Iter number: " << 1 << " out of " << m_num_mcmc_iters << "\n";
            std::cout << "***Iter number: " << 1 << " out of " << m_num_mcmc_iters << "\n";        
        
            // write accepted (initial) parameters to file (initial guesses are always "accepted")
            utils::logParams<numparams, float_t>(m_current_trans_theta, m_samples_file_stream);
            
            // get logLike (we use cancel token but it never changes) 
            std::atomic_bool cancel_token(false);
            if (!m_multicore){
                oldLogLike = logLikeEvaluate(m_current_trans_theta, m_data, cancel_token);
            }else{
                std::vector<std::future<float_t> > newLogLikes;
                for(size_t i = 0; i < m_num_extra_threads; ++i){
                    newLogLikes.push_back(std::async(std::launch::async,
                                                     &ada_pmmh::logLikeEvaluate,
                                                     this,
                                                     std::cref(m_current_trans_theta), 
                                                     std::cref(m_data),
                                                     std::ref(cancel_token)));               
                }
                for(size_t i = 0; i < m_num_extra_threads; ++i){
                    oldLogLike += newLogLikes[i].get();
                }
                oldLogLike /= m_num_extra_threads;
            }
            
            // store prior for next round
            oldLogPrior = logPriorEvaluate(m_current_trans_theta);
            if( std::isinf(oldLogPrior) || std::isnan(oldLogPrior)){
                std::cerr << "oldLogPrior must be a real number. returning.\n";
                return;
            }            
            
            // increase the iteration counter
            m_iter++;

        } 
        else { // not the first iteration      

            // update sample moments (with the parameters that were just accepted) and Ct so we can qSample()
            update_moments_and_Ct(m_current_trans_theta);
            m_message_stream << "Ct: \n " << get_ct() << "\n";
            
            // propose a new theta
            psv proposed_trans_theta = qSample(m_current_trans_theta);
            
            // store some densities                        
            float_t newLogPrior = logPriorEvaluate(proposed_trans_theta);
            float_t logQOldToNew = logQEvaluate(m_current_trans_theta, proposed_trans_theta);
            float_t logQNewToOld = logQEvaluate(proposed_trans_theta, m_current_trans_theta);
    
            // get the likelihood
            float_t newLL(0.0);
            std::atomic_bool cancel_token(false);
            if (!m_multicore){
                newLL = logLikeEvaluate(proposed_trans_theta, m_data, cancel_token);
            }else{
                std::vector<std::future<float_t> > newLogLikes;
                for(size_t i = 0; i < m_num_extra_threads; ++i){
                    newLogLikes.push_back(std::async(std::launch::async,
                                                     &ada_pmmh::logLikeEvaluate,
                                                     this,
                                                     std::cref(proposed_trans_theta), 
                                                     std::cref(m_data),
                                                     std::ref(cancel_token)));               
                }
                for(size_t i = 0; i < m_num_extra_threads; ++i){
                    newLL += newLogLikes[i].get();
                }
                newLL /= m_num_extra_threads;
            }

            // accept or reject proposal
            float_t logAR = newLogPrior + logQNewToOld + newLL - oldLogPrior - logQOldToNew - oldLogLike;                
                
            // output some stuff
            m_message_stream << "***Iter number: " << m_iter+1 << " out of " << m_num_mcmc_iters << "\n";
            std::cout << "***Iter number: " << m_iter+1 << " out of " << m_num_mcmc_iters << "\n";        

            m_message_stream << "acceptance rate: " << m_ma_accept_rate << " \n";
            std::cout << "acceptance rate: " << m_ma_accept_rate << " \n";            

            m_message_stream << "oldLogLike: " << oldLogLike << "\n";
            std::cout << "oldLogLike: " << oldLogLike << "\n";
            
            m_message_stream << "newLogLike: " << newLL << "\n";
            std::cout << "newLogLike: " << newLL << "\n";

            m_message_stream << "PriorRatio: " << std::exp(newLogPrior - oldLogPrior) << "\n";
            std::cout << "PriorRatio: " << std::exp(newLogPrior - oldLogPrior) << "\n";
            
            m_message_stream << "LikeRatio: " << std::exp(newLL - oldLogLike) << "\n";
            std::cout << "LikeRatio: " << std::exp(newLL - oldLogLike) << "\n";
            
            m_message_stream << "AR: " << std::exp(logAR) << "\n";
            std::cout << "AR: " << std::exp(logAR) << "\n";

            // decide whether to accept or reject
            float_t draw = runif.sample();
            if ( std::isinf(-logAR)){
                // 0 acceptance rate
                std::cout << "rejecting!\n";
                m_ma_accept_rate = 0.0/(m_iter+1.0) + m_iter*m_ma_accept_rate/(m_iter+1.0);
                // do not change the parameters
                // oldPrior stays the same 
                // oldLogLike stays the same
                m_iter++; // increase number of iter
                m_message_stream << "rejected 100 percent\n";
            }else if (logAR >= 0.0){
                // 100 percent accept 
                std::cout << "accepting!\n";
                m_ma_accept_rate = 1.0/(m_iter+1.0) + m_iter*m_ma_accept_rate/(m_iter+1.0);
                m_current_trans_theta = proposed_trans_theta;
                oldLogPrior = newLogPrior;
                oldLogLike = newLL;
                m_message_stream << "accepted 100 percent\n";
                m_iter++; // increase number of iters
            }else if ( std::log(draw) <= logAR ) {
                // probabilistic accept
                std::cout << "accepting!\n";
                m_iter++; // increase number of iters
                m_ma_accept_rate = 1.0/(m_iter+1.0) + m_iter*m_ma_accept_rate/(m_iter+1.0);
                m_current_trans_theta = proposed_trans_theta;
                oldLogPrior = newLogPrior;
                oldLogLike = newLL;
                m_message_stream << "accepted probabilistically\n";
            } else if ( std::log(draw) > logAR ) {
                std::cout << "rejecting!\n";
                // probabilistically reject
                // parameters do not change
                // oldPrior stays the same 
                // oldLogLike stays the same     
                m_ma_accept_rate = 0.0/(m_iter+1.0) + m_iter*m_ma_accept_rate/(m_iter+1.0);
                m_iter++; // increase number of iters           
                m_message_stream << "rejected probabilistically\n";
            }else if (std::isnan(logAR) ){ 
                // this is unexpected behavior
                m_iter++; // increase number of iters
                m_ma_accept_rate = 0.0/(m_iter+1.0) + m_iter*m_ma_accept_rate/(m_iter+1.0);
                std::cerr << "there was a NaN. Not accepting proposal. \n";
                //std::cerr << "newLogLike: " << newLogLike << "\n";
                //std::cerr << "oldLogLikeL " << oldLogLike << "\n";
                //std::cerr << "newLogPrior: " << newLogPrior << "\n";
                //std::cerr << "oldLogPrior: " << oldLogPrior << "\n";
                //std::cerr << "logQNewToOld: "<< logQNewToOld << "\n";
                //std::cerr << "logQOldToNew: " << logQOldToNew << "\n";
                // does not terminate!
                // parameters don't change
                // oldPrior stays the same 
                // oldLogLike stays the same  
            } else {
                // this case should never be triggered
                std::cerr << "you coded your MCMC incorrectly\n";
                std::cerr << "stopping...";
                m_iter++; // increase number of iters
            }
                
            // log the theta which may have changedor not
            utils::logParams<numparams,float_t>(m_current_trans_theta, m_samples_file_stream);
                
        } // else (not the first iteration)    
    } // while(m_iter < m_num_mcmc_iters) // every iteration
    
    // stop writing thetas and messages
    m_samples_file_stream.close();
    m_message_stream.close();
    
}



#endif //ADA_PMMH_H
