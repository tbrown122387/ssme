#ifndef ADA_PMMH_MVN_H
#define ADA_PMMH_MVN_H

#include <vector>
#include <Eigen/Dense>
#include <iostream> // ofstream
#include <thread> // hardware_concurrency
#include <atomic> // atomic_bool
#include <fstream> // ofstream
#include <future> // std::future
#include <chrono> //std::chrono::system_clock::now()

#include "rv_eval.h"
#include "rv_samp.h"
#include "utils.h" // readInData
#include "param_pack.h"


/**
 * @class ada_pmmh_mvn
 * @author t
 * @date 24/05/18
 * @file ada_pmmh_mvn.h
 * @brief Performs adaptive particle marginal metropolis-hastings sampling, using a 
 * multivariate normal distribution as the proposal. This samples on the transformed 
 * space, but it writes out the untransformed/constrained samples to the output. The   
 * priors requested by the user are for the (hopefully more convenient) un-transformed 
 * or constrained space. This means that the user never has to worry about handling any 
 * kind of Jacobian--just specify a prior on the familiar space, and a function that 
 * approximates log-likelihoods.
 */
template<size_t numparams, size_t dimobs, size_t numparts, typename float_t>
class ada_pmmh_mvn{
public:

    using osv = Eigen::Matrix<float_t,dimobs,1>;
    using psv = Eigen::Matrix<float_t,numparams,1>;
    using psm = Eigen::Matrix<float_t,numparams,numparams>;
    
    /**
     * @brief The constructor
     * @param start_trans_theta the initial transformed parameters you want to start sampling from.
     * @param num_mcmc_iters the number of MCMC iterations you want to do.
     * @param data_file the location of the observed time series data (input).
     * @param samples_file the location where you want to store the theta samples (output).
     * @param messages_file the location where you want to store the messages (output).
     * @param mc stands for multicore. true or false if you want to use extra cores.
     * @param t0 time you start adapting
     * @param t1 time you stop adapting
     * @param C0 initial covariance matrix for proposal distribution.
     * @param print_to_console true if you want to see messages in real time
     * @param print_every_k print messages and samples every (this number) iterations
     */
    ada_pmmh_mvn(const psv &start_trans_theta, 
                 const std::vector<TransType>& tts,
                 const unsigned int &num_mcmc_iters,
                 const std::string &data_file, // TODO: describe formatting rules (e.g. column orders, column names, etc.)
                 const std::string &sample_file_base_name,
                 const std::string &message_file_base_name,
                 const bool &mc,
                 const unsigned int &t0,
                 const unsigned int &t1,
                 const psm &C0,
                 bool print_to_console,
                 unsigned int print_every_k);
             
    
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
     * @brief Evaluates the log of the model's prior distribution assuming the original/nontransformed/contrained parameterization
     * @param theta the parameters argument (nontransformed/constrained parameterization). 
     * @return the log of the prior density.
     */
    virtual float_t logPriorEvaluate(const paramPack<float_t>& theta) = 0;


    /**
     * @brief Evaluates (approximates) the log-likelihood with a particle filter.
     * @param theta the parameters with which to run the particle filter.
     * @param data the observed data with which to run the particle filter.
     * @return the evaluation (as a float_t) of the log likelihood approximation.
     */
    virtual float_t logLikeEvaluate(const paramPack<float_t>& theta, const std::vector<osv> &data) = 0;
    
             
private:
    std::vector<osv> m_data;
    paramPack<float_t> m_current_theta;
    std::vector<TransType> m_tts;
    psm m_sigma_hat; // for transformed parameters; n-1 in the denominator.
    psv m_mean_trans_theta;
    float_t m_ma_accept_rate;
    unsigned int m_t0;  // the time it starts adapting
    unsigned int m_t1; // the time it stops adapting
    psm m_Ct;
    rvsamp::MVNSampler<numparams,float_t> m_mvn_gen;
    std::ofstream m_samples_file_stream; 
    std::ofstream m_message_stream;
    unsigned int m_num_mcmc_iters;
    unsigned int m_num_extra_threads;
    bool m_multicore;
    unsigned int m_iter; // current iter
    float_t m_sd; // perhaps there's a better name for this
    float_t m_eps;
    bool m_print_to_console;
    unsigned int m_print_every_k;    
    
    
    void update_moments_and_Ct(const paramPack<float_t>& newTheta);
    psv qSample(const paramPack<float_t>& oldTheta);
    float_t logQEvaluate(const paramPack<float_t>& oldParams, const paramPack<float_t>& newParams);
};


template<size_t numparams, size_t dimobs, size_t numparts, typename float_t>
ada_pmmh_mvn<numparams,dimobs,numparts,float_t>::ada_pmmh_mvn(
                                            const psv &start_trans_theta, 
                                            const std::vector<TransType>& tts,
                                            const unsigned int &num_mcmc_iters,
                                            const std::string &data_file, 
                                            const std::string &sample_file_base_name,
                                            const std::string &message_file_base_name,
                                            const bool &mc,
                                            const unsigned int &t0,
                                            const unsigned int &t1,
                                            const psm &C0,
                                            bool print_to_console,
                                            unsigned int print_every_k)
 : m_current_theta(start_trans_theta, tts)
 , m_tts(tts)
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
 , m_print_to_console(print_to_console)
 , m_print_every_k(print_every_k)
{
    m_data = utils::readInData<dimobs,float_t>(data_file);
    std::string samples_file = utils::genStringWithTime(sample_file_base_name);
    m_samples_file_stream.open(samples_file); 
    std::string messages_file = utils::genStringWithTime(message_file_base_name);
    m_message_stream.open(messages_file);  
    m_num_extra_threads = std::thread::hardware_concurrency() - 1;
}


template<size_t numparams, size_t dimobs, size_t numparts, typename float_t>
void ada_pmmh_mvn<numparams,dimobs,numparts,float_t>::update_moments_and_Ct(const paramPack<float_t>& newTheta)  
{
    // if m_iter = 1, that means we're on iteration 2, 
    // but we're calling this based on the previous iteration, 
    // so that's iteration 1, think of n as the actual 
    // number of iterations that have happened then (not counting from 0)
    // On the other hand, yu might not want to count the first iteration, 
    // though, because we didn't "propose" anything and probabilistically accept/reject it
    psv newTransTheta = newTheta.getTransParams();
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
auto ada_pmmh_mvn<numparams,dimobs,numparts,float_t>::get_ct() const -> psm
{
    return m_Ct;
}


template<size_t numparams, size_t dimobs, size_t numparts, typename float_t>
auto ada_pmmh_mvn<numparams,dimobs,numparts,float_t>::qSample(const paramPack<float_t>& oldParams) -> psv
{
    // assumes that Ct has already been updated
    // recall that we are sampling on the transformed/unconstrained space    
    psv oldTransParams = oldParams.getTransParams();
    m_mvn_gen.setMean(oldTransParams);
    m_mvn_gen.setCovar(m_Ct);
    return m_mvn_gen.sample();
}


template<size_t numparams, size_t dimobs, size_t numparts, typename float_t>
void ada_pmmh_mvn<numparams,dimobs,numparts,float_t>::commenceSampling()
{

    // random number stuff to decide on whether to accept or reject
    rvsamp::UniformSampler<float_t> runif; 
    
    float_t oldLogLike (0.0);
    float_t oldLogPrior(0.0);
    while(m_iter < m_num_mcmc_iters) // every iteration
    {        

        // first iteration no acceptance probability
        if (m_iter == 0) { 
            
            if(m_iter % m_print_every_k == 0){
                
                m_message_stream << "***Iter number: " << 1 << " out of " << m_num_mcmc_iters << "\n";
            

                if(m_print_to_console)
                    std::cout << "***Iter number: " << 1 << " out of " << m_num_mcmc_iters << "\n";        
                         
                // write accepted (initial) parameters to file (initial guesses are always "accepted")
                // notice that they are the untransformed/constrained versions!
                utils::logParams<numparams,float_t>(m_current_theta.getUnTransParams(), m_samples_file_stream);

            }
           
            // get logLike 
            if (!m_multicore){
                oldLogLike = logLikeEvaluate(m_current_theta, m_data);
            }else{
                std::vector<std::future<float_t> > newLogLikes;
                for(size_t i = 0; i < m_num_extra_threads; ++i){
                    newLogLikes.push_back(std::async(std::launch::async,
                                                     &ada_pmmh_mvn::logLikeEvaluate,
                                                     this,
                                                     std::cref(m_current_theta), 
                                                     std::cref(m_data)));               
                }
                for(size_t i = 0; i < m_num_extra_threads; ++i){
                    oldLogLike += newLogLikes[i].get();
                }
                oldLogLike /= m_num_extra_threads;
            }
            
            // store prior for next round
            oldLogPrior = logPriorEvaluate(m_current_theta) + m_current_theta.getLogJacobian(); 
            if( std::isinf(oldLogPrior) ){
                std::cerr << "oldLogPrior is infinite. Returning from commenceSampling().\n";
                return;
            }
            if(std::isnan(oldLogPrior)){
                std::cerr << "oldLogPrior is not a number. Returning from commenceSampling().\n";
                return;
            }
            
            // increase the iteration counter
            m_iter++;

        } 
        else { // not the first iteration      

            // update sample moments (with the parameters that were just accepted) and Ct so we can qSample()
            update_moments_and_Ct(m_current_theta);
            
            // propose a new theta
            psv proposed_trans_theta = qSample(m_current_theta);
            paramPack<float_t> proposed_theta(proposed_trans_theta, m_tts);
            
            // store some densities                        
            float_t newLogPrior = logPriorEvaluate(proposed_theta) + proposed_theta.getLogJacobian();
    
            // get the likelihood
            float_t newLL(0.0);
            if (!m_multicore){
                newLL = logLikeEvaluate(proposed_theta, m_data);
            }else{
                std::vector<std::future<float_t> > newLogLikes;
                for(size_t i = 0; i < m_num_extra_threads; ++i){
                    newLogLikes.push_back(std::async(std::launch::async,
                                                     &ada_pmmh_mvn::logLikeEvaluate,
                                                     this,
                                                     std::cref(proposed_theta), 
                                                     std::cref(m_data)));               
                }
                for(size_t i = 0; i < m_num_extra_threads; ++i){
                    newLL += newLogLikes[i].get();
                }
                newLL /= m_num_extra_threads;
            }

            // accept or reject proposal (assumes multivariate normal proposal which means it's symmetric)
            float_t logAR = newLogPrior + newLL - oldLogPrior - oldLogLike;                
                
            // output some stuff
            if(m_iter % m_print_every_k == 0){
                 
                m_message_stream << "***Iter number: " << m_iter+1 << " out of " << m_num_mcmc_iters << "\n"
                                  << "acceptance rate: " << m_ma_accept_rate << " \n"
                                  << "oldLogLike: " << oldLogLike << "\n"
                                  << "newLogLike: " << newLL << "\n"
                                  << "PriorRatio: " << std::exp(newLogPrior - oldLogPrior) << "\n"
                                  << "LikeRatio: " << std::exp(newLL - oldLogLike) << "\n"
                                  << "AR: " << std::exp(logAR) << "\n"
                                  << "Ct: \n " << get_ct() << "\n";                
            
                if(m_print_to_console){

                    std::cout        << "***Iter number: " << m_iter+1 << " out of " << m_num_mcmc_iters << "\n"
                                     << "acceptance rate: " << m_ma_accept_rate << " \n"
                                     << "oldLogLike: " << oldLogLike << "\n"
                                     << "newLogLike: " << newLL << "\n"
                                     << "PriorRatio: " << std::exp(newLogPrior - oldLogPrior) << "\n"
                                     << "LikeRatio: " << std::exp(newLL - oldLogLike) << "\n"
                                     << "AR: " << std::exp(logAR) << "\n"
                                     << "Ct: \n " << get_ct() << "\n";                

                }
            }

            // decide whether to accept or reject
            float_t draw = runif.sample();
            if ( std::isinf(-logAR)){  // 0 acceptance rate
                
                if(m_iter % m_print_every_k == 0){
                    
                    m_message_stream << "rejected 100 percent\n";
             
                    // log the theta which may have changedor not
                    // notice that this is on the untransformed or constrained space
                    utils::logParams<numparams,float_t>(m_current_theta.getUnTransParams(), m_samples_file_stream);

                    if(m_print_to_console)
                        std::cout << "rejected 100 percent\n";

                }

                m_ma_accept_rate = 0.0/(m_iter+1.0) + m_iter*m_ma_accept_rate/(m_iter+1.0);
                // do not change the parameters
                // oldPrior stays the same 
                // oldLogLike stays the same
                m_iter++; // increase number of iter
            
            }else if (logAR >= 0.0){ // 100 percent accept 

                if(m_iter % m_print_every_k == 0){
                    
                    m_message_stream << "accepted 100 percent\n";
             
                    // log the theta which may have changedor not
                    // notice that this is on the untransformed or constrained space
                    utils::logParams<numparams,float_t>(m_current_theta.getUnTransParams(), m_samples_file_stream);

                    if(m_print_to_console)
                        std::cout << "accepted 100 percent\n";

                }

                m_ma_accept_rate = 1.0/(m_iter+1.0) + m_iter*m_ma_accept_rate/(m_iter+1.0);
                m_current_theta.takeValues(proposed_theta);
                oldLogPrior = newLogPrior;
                oldLogLike = newLL;
                m_iter++; // increase number of iters

            }else if ( std::log(draw) <= logAR ) { // probabilistic accept

                if(m_iter % m_print_every_k == 0){
                    
                    m_message_stream << "accepted probabilistically\n";
             
                    // log the theta which may have changedor not
                    // notice that this is on the untransformed or constrained space
                    utils::logParams<numparams,float_t>(m_current_theta.getUnTransParams(), m_samples_file_stream);

                    if(m_print_to_console)
                        std::cout << "accepted probabilistically\n";

                }

                m_iter++; // increase number of iters
                m_ma_accept_rate = 1.0/(m_iter+1.0) + m_iter*m_ma_accept_rate/(m_iter+1.0);
                m_current_theta.takeValues(proposed_theta);
                oldLogPrior = newLogPrior;
                oldLogLike = newLL;

            } else if ( std::log(draw) > logAR ) {

                if(m_iter % m_print_every_k == 0){
                    
                    m_message_stream << "rejected probabilistically\n";
             
                    // log the theta which may have changedor not
                    // notice that this is on the untransformed or constrained space
                    utils::logParams<numparams,float_t>(m_current_theta.getUnTransParams(), m_samples_file_stream);

                    if(m_print_to_console)
                        std::cout << "rejected probabilistically\n";

                }

                // probabilistically reject
                // parameters do not change
                // oldPrior stays the same 
                // oldLogLike stays the same     
                m_ma_accept_rate = 0.0/(m_iter+1.0) + m_iter*m_ma_accept_rate/(m_iter+1.0);
                m_iter++; // increase number of iters           

            }else if (std::isnan(logAR) ){ 

                if(m_iter % m_print_every_k == 0){
                    
                    m_message_stream << "rejected from NaN\n";
             
                    // log the theta which may have changedor not
                    // notice that this is on the untransformed or constrained space
                    utils::logParams<numparams,float_t>(m_current_theta.getUnTransParams(), m_samples_file_stream);

                    if(m_print_to_console)
                        std::cout << "rejected from NaN\n";

                }

                // this is unexpected behavior
                m_iter++; // increase number of iters
                m_ma_accept_rate = 0.0/(m_iter+1.0) + m_iter*m_ma_accept_rate/(m_iter+1.0);
                std::cerr << "there was a NaN. Not accepting proposal. \n";

                // does not terminate!
                // parameters don't change
                // oldPrior stays the same 
                // oldLogLike stays the same  
            } else {
                // this case should never be triggered
                std::cerr << "you coded your MCMC incorrectly\n";
                std::cerr << "stopping...";
                m_iter++; // increase number of iters
		return;
            }
                
               
        } // else (not the first iteration)    
    } // while(m_iter < m_num_mcmc_iters) // every iteration
    
    // stop writing thetas and messages
    m_samples_file_stream.close();
    m_message_stream.close();
    
}


#endif //ADA_PMMH_MVN_H