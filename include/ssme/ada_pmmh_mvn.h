#ifndef ADA_PMMH_MVN_H
#define ADA_PMMH_MVN_H

#include <vector>
#include <Eigen/Dense>
#include <iostream> // ofstream
//#include <thread> // hardware_concurrency
#include <fstream> // ofstream
#include <chrono> //std::chrono::system_clock::now()

#include <pf/rv_eval.h>
#include <pf/rv_samp.h>
#include <ssme/utils.h> // readInData
#include <ssme/parameters.h>


/**
 * @class ada_pmmh_mvn
 * @author Taylor
 * @file ada_pmmh_mvn.h
 * @brief Performs (possibly-adaptive) particle marginal Metropolis-Hastings sampling, using a 
 * multivariate normal distribution as the parameter proposal. This samples on the transformed 
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
     * @brief Constructs algorithm object
     * @param start_trans_theta the initial transformed parameters you want to start sampling from.
     * @param num_mcmc_iters the number of MCMC iterations you want to do.
     * @param data_file the location of the observed time series data (input).
     * @param samples_file the location where you want to store the theta samples (output).
     * @param messages_file the location where you want to store the messages (output).
     * @param mc stands for multicore. true or false if you want to use extra cores.
     * @param t0 iteration you start adapting
     * @param t1 iteration you stop adapting
     * @param C0 initial covariance matrix for proposal distribution.
     * @param print_to_console true if you want to see messages in real time
     * @param print_every_k print messages and samples every (this number) iterations
     */
    ada_pmmh_mvn(const psv &start_trans_theta, 
                 const std::vector<param::trans_type>& tts,
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
    void commence_sampling();
   

    /**
     * @brief Evaluates the log of the model's prior distribution assuming the original/nontransformed/contrained parameterization
     * @param theta the parameters argument (nontransformed/constrained parameterization). 
     * @return the log of the prior density.
     */
    virtual float_t log_prior_eval(const param::pack<float_t>& theta) = 0;


    /**
     * @brief Evaluates (approximates) the log-likelihood with a particle filter.
     * @param theta the parameters with which to run the particle filter.
     * @param data the observed data with which to run the particle filter.
     * @return the evaluation (as a float_t) of the log likelihood approximation.
     */
    virtual float_t log_like_eval(const param::pack<float_t>& theta, 
                                  const std::vector<osv> &data) = 0;
    
             
private:
    std::vector<osv> m_data;
    param::pack<float_t> m_current_theta;
    std::vector<param::trans_type> m_tts;
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
//    unsigned int m_num_extra_threads;
    bool m_multicore;
    unsigned int m_iter; // current iter
    float_t m_sd; // perhaps there's a better name for this
    float_t m_eps;
    bool m_print_to_console;
    unsigned int m_print_every_k;    
    
    /* thread pool (its function can only take one parameter) */
    struct param_and_data{ 
        param::pack<float_t> params; 
        std::vector<osv> data;
    };
    using t_pool = thread_pool<param_and_data, float_t>;
    t_pool::F m_f; 
    t_pool m_pool; 
    
    void update_moments_and_Ct(const param::pack<float_t>& new_theta);
    psv q_samp(const param::pack<float_t>& oldTheta);
    float_t log_q_eval(const param::pack<float_t>& oldParams, const param::pack<float_t>& newParams);
};


template<size_t numparams, size_t dimobs, size_t numparts, typename float_t>
ada_pmmh_mvn<numparams,dimobs,numparts,float_t>::ada_pmmh_mvn(
                                            const psv &start_trans_theta, 
                                            const std::vector<param::trans_type>& tts,
                                            const unsigned int &num_mcmc_iters,
                                            const unsigned int &num_pfilters,
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
 , m_f([this](param_and_data both) -> float_t { 
         return this->log_like_eval(both.params, both.data); 
         })
 , m_pool(m_f, num_pfilters, mc)
{
    m_data = utils::read_in_data<dimobs,float_t>(data_file);
    std::string samples_file = utils::gen_string_with_time(sample_file_base_name);
    m_samples_file_stream.open(samples_file); 
    std::string messages_file = utils::gen_string_with_time(message_file_base_name);
    //m_message_stream.open(messages_file); // TODO does this open upon constrution by default? 
}


template<size_t numparams, size_t dimobs, size_t numparts, typename float_t>
void ada_pmmh_mvn<numparams,dimobs,numparts,float_t>::update_moments_and_Ct(const param::pack<float_t>& new_theta)  
{
    // if m_iter = 1, that means we're on iteration 2, 
    // but we're calling this based on the previous iteration, 
    // so that's iteration 1, think of n as the actual 
    // number of iterations that have happened then (not counting from 0)
    // On the other hand, yu might not want to count the first iteration, 
    // though, because we didn't "propose" anything and probabilistically accept/reject it
    psv new_trans_theta = new_theta.get_trans_params();
    if(m_iter == 1){
        // at the moment m_meanTransTheta is zero, so we add m_currentTransTheta
        // we'll also leave m_sigma_hat to be zero for the time being
        m_mean_trans_theta += new_trans_theta;
    }else if(m_iter == 2){
        // at the beginning of this call, m_mean_trans_theta is x1*x1^T and m_sigma_hat is zero
        // this is a an algebraically simplified version of the split up non-recursive formula
        m_sigma_hat = m_mean_trans_theta*m_mean_trans_theta.transpose() 
                    + new_trans_theta * new_trans_theta.transpose() 
                    - m_mean_trans_theta * new_trans_theta.transpose() 
                    - new_trans_theta * m_mean_trans_theta.transpose();
        m_sigma_hat *= .5;
        m_mean_trans_theta = .5* m_mean_trans_theta + .5*new_trans_theta;     
    }else if( m_iter > 2){
        // regular formula. see
        // https://stats.stackexchange.com/questions/310680/sequential-recursive-online-calculation-of-sample-covariance-matrix/310701#310701
        m_sigma_hat = m_sigma_hat*(m_iter-2.0)/(m_iter-1.0) 
                    + (new_trans_theta - m_mean_trans_theta)*(new_trans_theta - m_mean_trans_theta).transpose()/m_iter;
        m_mean_trans_theta = ((m_iter-1.0)*m_mean_trans_theta + new_trans_theta)/m_iter;

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
auto ada_pmmh_mvn<numparams,dimobs,numparts,float_t>::q_samp(const param::pack<float_t>& old_params) -> psv
{
    // assumes that Ct has already been updated
    // recall that we are sampling on the transformed/unconstrained space    
    psv old_trans_params = old_params.get_trans_params();
    m_mvn_gen.setMean(old_trans_params);
    m_mvn_gen.setCovar(m_Ct);
    return m_mvn_gen.sample();
}


template<size_t numparams, size_t dimobs, size_t numparts, typename float_t>
void ada_pmmh_mvn<numparams,dimobs,numparts,float_t>::commence_sampling()
{

    // random number stuff to decide on whether to accept or reject
    rvsamp::UniformSampler<float_t> runif; 
    
    float_t old_log_like (0.0);
    float_t old_log_prior(0.0);
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
                utils::log_params<numparams,float_t>(m_current_theta.get_untrans_params(), 
                                                     m_samples_file_stream);

            }
           
            // get logLike 
            old_log_like = m_pool.work( param_and_data{m_current_theta, m_data } );
                                       
            // store prior for next round
            old_log_prior = log_prior_eval(m_current_theta) + m_current_theta.get_log_jacobian(); 
            if( std::isinf(old_log_prior) ){
                std::cerr << "old_log_prior is infinite. Returning from commenceSampling().\n";
                return;
            }
            if(std::isnan(old_log_prior)){
                std::cerr << "old_log_prior is not a number. Returning from commenceSampling().\n";
                return;
            }
            
            // increase the iteration counter
            m_iter++;

        } 
        else { // not the first iteration      

            // update sample moments (with the parameters that were just accepted) and Ct so we can q_samp()
            update_moments_and_Ct(m_current_theta);
            
            // propose a new theta
            psv proposed_trans_theta = q_samp(m_current_theta);
            param::pack<float_t> proposed_theta(proposed_trans_theta, m_tts);
            
            // store some densities                        
            float_t new_log_prior = logPriorEvaluate(proposed_theta) + proposed_theta.getLogJacobian();
    
            // get the likelihood
            new_log_like = m_pool.work( param_and_data{proposed_theta, m_data } );

            // accept or reject proposal (assumes multivariate normal proposal which means it's symmetric)
            float_t logAR = new_log_prior + new_log_like - old_log_prior - old_log_like;                
                
            // output some stuff
            if(m_iter % m_print_every_k == 0){
                 
                m_message_stream << "***Iter number: " << m_iter+1 << " out of " << m_num_mcmc_iters << "\n"
                                  << "acceptance rate: " << m_ma_accept_rate << " \n"
                                  << "old_log_like: " << old_log_like << "\n"
                                  << "newLogLike: " << new_log_like << "\n"
                                  << "PriorRatio: " << std::exp(new_log_prior - old_log_prior) << "\n"
                                  << "LikeRatio: " << std::exp(new_log_like - old_log_like) << "\n"
                                  << "AR: " << std::exp(logAR) << "\n"
                                  << "Ct: \n " << get_ct() << "\n";                
            
                if(m_print_to_console){

                    std::cout        << "***Iter number: " << m_iter+1 << " out of " << m_num_mcmc_iters << "\n"
                                     << "acceptance rate: " << m_ma_accept_rate << " \n"
                                     << "old_log_like: " << old_log_like << "\n"
                                     << "newLogLike: " << new_log_like << "\n"
                                     << "PriorRatio: " << std::exp(new_log_prior - old_log_prior) << "\n"
                                     << "LikeRatio: " << std::exp(new_log_like - old_log_like) << "\n"
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
                    utils::log_params<numparams,float_t>(m_current_theta.get_untrans_params(), m_samples_file_stream);

                    if(m_print_to_console)
                        std::cout << "rejected 100 percent\n";

                }

                m_ma_accept_rate = 0.0/(m_iter+1.0) + m_iter*m_ma_accept_rate/(m_iter+1.0);
                // do not change the parameters
                // oldPrior stays the same 
                // old_log_like stays the same
                m_iter++; // increase number of iter
            
            }else if (logAR >= 0.0){ // 100 percent accept 

                if(m_iter % m_print_every_k == 0){
                    
                    m_message_stream << "accepted 100 percent\n";
             
                    // log the theta which may have changedor not
                    // notice that this is on the untransformed or constrained space
                    utils::log_params<numparams,float_t>(m_current_theta.get_untrans_params(), m_samples_file_stream);

                    if(m_print_to_console)
                        std::cout << "accepted 100 percent\n";

                }

                m_ma_accept_rate = 1.0/(m_iter+1.0) + m_iter*m_ma_accept_rate/(m_iter+1.0);
                m_current_theta.takeValues(proposed_theta);
                old_log_prior = new_log_prior;
                old_log_like = new_log_like;
                m_iter++; // increase number of iters

            }else if ( std::log(draw) <= logAR ) { // probabilistic accept

                if(m_iter % m_print_every_k == 0){
                    
                    m_message_stream << "accepted probabilistically\n";
             
                    // log the theta which may have changedor not
                    // notice that this is on the untransformed or constrained space
                    utils::log_params<numparams,float_t>(m_current_theta.get_untrans_params(), m_samples_file_stream);

                    if(m_print_to_console)
                        std::cout << "accepted probabilistically\n";

                }

                m_iter++; // increase number of iters
                m_ma_accept_rate = 1.0/(m_iter+1.0) + m_iter*m_ma_accept_rate/(m_iter+1.0);
                m_current_theta.take_values(proposed_theta);
                old_log_prior = new_log_prior;
                old_log_like = new_log_like;

            } else if ( std::log(draw) > logAR ) {

                if(m_iter % m_print_every_k == 0){
                    
                    m_message_stream << "rejected probabilistically\n";
             
                    // log the theta which may have changedor not
                    // notice that this is on the untransformed or constrained space
                    utils::log_params<numparams,float_t>(m_current_theta.get_untrans_params(), m_samples_file_stream);

                    if(m_print_to_console)
                        std::cout << "rejected probabilistically\n";

                }

                // probabilistically reject
                // parameters do not change
                // oldPrior stays the same 
                // old_log_like stays the same     
                m_ma_accept_rate = 0.0/(m_iter+1.0) + m_iter*m_ma_accept_rate/(m_iter+1.0);
                m_iter++; // increase number of iters           

            }else if (std::isnan(logAR) ){ 

                if(m_iter % m_print_every_k == 0){
                    
                    m_message_stream << "rejected from NaN\n";
             
                    // log the theta which may have changedor not
                    // notice that this is on the untransformed or constrained space
                    utils::log_params<numparams,float_t>(m_current_theta.get_untrans_params(), m_samples_file_stream);

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
                // old_log_like stays the same  
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
