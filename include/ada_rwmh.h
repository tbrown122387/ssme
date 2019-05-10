#ifndef ADA_RWMH_H
#define ADA_RWMH_H

#include <vector>
#include <Eigen/Dense>
#include <iostream> // ofstream
#include <fstream> // ofstream
#include <chrono> //std::chrono::system_clock::now()
//#include <typeinfo> // typeid()

#include "rv_eval.h"
#include "rv_samp.h"
#include "utils.h" // readInData
#include "param_pack.h"


/**
 * @class ada_rwmh
 * @author t
 * @file ada_rwmh.h
 * @brief Performs adaptive random walk metropolis-hastings sampling (uses a 
 * multivariate normal distribution as the proposal). This samples on the transformed 
 * space, but it writes the untransformed/constrained samples to the output. The   
 * priors requested by the user are for the (more convenient) un-transformed 
 * aka constrained space. This means that the user never has to worry about handling any 
 * kind of Jacobian.
 */
template<size_t numparams, size_t dimobs>
class ada_rwmh{
public:

    /* type aliases */
    using osv = Eigen::Matrix<double,dimobs,1>;
    using psv = Eigen::Matrix<double,numparams,1>;
    using psm = Eigen::Matrix<double,numparams,numparams>;
    

    /**
     * @brief The constructor
     * @param start_trans_theta the initial transformed parameters you want to start sampling from.
     * @param num_mcmc_iters the number of MCMC iterations you want to do.
     * @param data_file the location of the observed time series data (input).
     * @param samples_file the location where you want to store the theta samples (output).
     * @param messages_file the location where you want to store the messages (output).
     * @param t0 time you start adapting
     * @param t1 time you stop adapting
     * @param C0 initial covariance matrix for proposal distribution.
     */
    ada_rwmh(
                 const psv &start_trans_theta, 
                 const std::vector<TransType>& tts,
                 const unsigned int &num_mcmc_iters,
                 const std::string &data_file, // TODO: describe formatting rules (e.g. column orders, column names, etc.)
                 const std::string &sample_file_base_name,
                 const std::string &message_file_base_name,
                 const unsigned int &t0,
                 const unsigned int &t1,
                 const psm &C0,
                 bool print_to_console);
             
    
    /**
     * @brief Get the current proposal distribution's covariance matrix.
     * @return the covariance matrix of q(theta' | theta)
     */
    psm get_ct() const;
    

    /**
     * @brief starts the sampling
     */
    void commenceSampling(const std::vector<Eigen::Matrix<double,dimobs,1>> &data);
    
    /**
     * @brief Evaluates the log of the model's prior distribution assuming the original/nontransformed/contrained parameterization
     * @param theta the parameters argument (nontransformed/constrained parameterization). 
     * @return the log of the prior density.
     */
    virtual double logPriorEvaluate(const paramPack& theta) = 0;


    /**
     * @brief Evaluates the log-likelihood.
     * @param theta the parameters of your likelihood.
     * @param data the observed data you're modeling.
     * @return the evaluation (as a double) of the log likelihood.
     */
    virtual double logLikeEvaluate(const paramPack& theta, const std::vector<osv> &data) = 0;
    
             
private:
    
    paramPack m_current_theta;
    std::vector<TransType> m_tts;
    psm m_sigma_hat; // for transformed parameters; n-1 in the denominator.
    psv m_mean_trans_theta;
    double m_ma_accept_rate;
    unsigned int m_t0;  // the time it starts adapting
    unsigned int m_t1; // the time it stops adapting
    psm m_Ct;
    rvsamp::MVNSampler<numparams> m_mvn_gen;
    std::ofstream m_samples_file_stream; 
    std::ofstream m_message_stream;
    unsigned int m_num_mcmc_iters;
    //unsigned int m_num_extra_threads;
    //bool m_multicore;
    unsigned int m_iter; // current iter
    double m_sd; // perhaps there's a better name for this
    double m_eps;
    bool m_print_to_console;
    
    
    
    void update_moments_and_Ct(const paramPack& newTheta);
    psv qSample(const paramPack& oldTheta);
    double logQEvaluate(const paramPack& oldParams, const paramPack& newParams);
};


template<size_t numparams, size_t dimobs>
ada_rwmh<numparams,dimobs>::ada_rwmh(
                                            const psv &start_trans_theta, 
                                            const std::vector<TransType>& tts,
                                            const unsigned int &num_mcmc_iters,
                                            const std::string &data_file, 
                                            const std::string &sample_file_base_name,
                                            const std::string &message_file_base_name,
                                            const unsigned int &t0,
                                            const unsigned int &t1,
                                            const psm &C0,
                                            bool print_to_console)
 : m_current_theta(start_trans_theta, tts)
 , m_tts(tts)
 , m_sigma_hat(psm::Zero())
 , m_mean_trans_theta(psv::Zero())
 , m_ma_accept_rate(0.0)
 , m_t0(t0), m_t1(t1)
 , m_Ct(C0)
 , m_num_mcmc_iters(num_mcmc_iters)
 , m_iter(0)
 , m_sd(2.4*2.4/numparams)
 , m_eps(.01)
 , m_print_to_console(print_to_console)
{
    std::string samples_file = utils::genStringWithTime(sample_file_base_name);
    m_samples_file_stream.open(samples_file); 
    std::string messages_file = utils::genStringWithTime(message_file_base_name);
    m_message_stream.open(messages_file);  
}


template<size_t numparams, size_t dimobs>
void ada_rwmh<numparams,dimobs>::update_moments_and_Ct(const paramPack& newTheta)  
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


template<size_t numparams, size_t dimobs>
auto ada_rwmh<numparams,dimobs>::get_ct() const -> psm
{
    return m_Ct;
}


template<size_t numparams, size_t dimobs>
auto ada_rwmh<numparams,dimobs>::qSample(const paramPack& oldParams) -> psv
{
    // assumes that Ct has already been updated
    // recall that we are sampling on the transformed/unconstrained space    
    psv oldTransParams = oldParams.getTransParams();
    m_mvn_gen.setMean(oldTransParams);
    m_mvn_gen.setCovar(m_Ct);
    return m_mvn_gen.sample();
}


template<size_t numparams, size_t dimobs>
void ada_rwmh<numparams,dimobs>::commenceSampling(const std::vector<Eigen::Matrix<double,dimobs,1>> &data)
{

    // random number stuff to decide on whether to accept or reject
    rvsamp::UniformSampler runif; 
    
    double oldLogLike (0.0);
    double oldLogPrior(0.0);
    while(m_iter < m_num_mcmc_iters) // every iteration
    {        

        // first iteration no acceptance probability
        if (m_iter == 0) { 
            
            m_message_stream << "***Iter number: " << 1 << " out of " << m_num_mcmc_iters << "\n";
            if(m_print_to_console)
                std::cout << "***Iter number: " << 1 << " out of " << m_num_mcmc_iters << "\n";        
        
            // write initial parameters to file (initial guesses are always "accepted")
            // notice that they are the untransformed/constrained versions!
            // this makes analysis easier when the time comes
            utils::logParams<numparams>(m_current_theta.getUnTransParams(), m_samples_file_stream);
            
            // get logLike 
            oldLogLike = logLikeEvaluate(m_current_theta, data);
            
            // store prior for next round
            oldLogPrior = logPriorEvaluate(m_current_theta) + m_current_theta.getLogJacobian(); ///!!!!!
            if( std::isinf(oldLogPrior) || std::isnan(oldLogPrior)){
                std::cerr << "oldLogPrior must be a real number. returning.\n";
                return;
            }            
            
            // increase the iteration counter
            m_iter++;

        } 
        else { // not the first iteration      

            // update sample moments (with the parameters that were just accepted) and Ct so we can qSample()
            update_moments_and_Ct(m_current_theta);
            m_message_stream << "Ct: \n " << get_ct() << "\n";                
            
            // propose a new theta
            psv proposed_trans_theta = qSample(m_current_theta);
            paramPack proposed_theta(proposed_trans_theta, m_tts);
            
            // store some densities                        
            double newLogPrior = logPriorEvaluate(proposed_theta) + proposed_theta.getLogJacobian();
    
            // get the likelihood
            double newLL = logLikeEvaluate(proposed_theta, data);

            // accept or reject proposal (assumes multivariate normal proposal which means it's symmetric)
            double logAR = newLogPrior + newLL - oldLogPrior - oldLogLike;                
                
            // output some stuff
            m_message_stream << "***Iter number: "  << m_iter+1                            << " out of " << m_num_mcmc_iters << "\n"
                             << "acceptance rate: " << m_ma_accept_rate                    << " \n"
                             << "oldLogLike: "      << oldLogLike                          << "\n"
                             << "newLogLike: "      << newLL                               << "\n"
                             << "PriorRatio: "      << std::exp(newLogPrior - oldLogPrior) << "\n"
                             << "LikeRatio: "       << std::exp(newLL - oldLogLike)        << "\n"
                             << "AR: "              << std::exp(logAR)                     << "\n";
            
            if(m_print_to_console){
                std::cout << "***Iter number: "  << m_iter+1                            << " out of " << m_num_mcmc_iters << "\n"
                          << "acceptance rate: " << m_ma_accept_rate                    << " \n"
                          << "oldLogLike: "      << oldLogLike                          << "\n"
                          << "newLogLike: "      << newLL                               << "\n"
                          << "PriorRatio: "      << std::exp(newLogPrior - oldLogPrior) << "\n"
                          << "LikeRatio: "       << std::exp(newLL - oldLogLike)        << "\n"
                          << "AR: "              << std::exp(logAR)                     << "\n";
 
            }

            // decide whether to accept or reject
            double draw = runif.sample();
            if ( std::isinf(-logAR)){
                // 0 acceptance rate
                if(m_print_to_console)
                    std::cout << "rejecting!\n";
                m_ma_accept_rate = 0.0/(m_iter+1.0) + m_iter*m_ma_accept_rate/(m_iter+1.0);
                // do not change the parameters
                // oldPrior stays the same 
                // oldLogLike stays the same
                m_iter++; // increase number of iter
                m_message_stream << "rejected 100 percent\n";
            }else if (logAR >= 0.0){
                // 100 percent accept 
                if(m_print_to_console)
                    std::cout << "accepted 100 percent\n";
                m_ma_accept_rate = 1.0/(m_iter+1.0) + m_iter*m_ma_accept_rate/(m_iter+1.0);
                m_current_theta.takeValues(proposed_theta);
                oldLogPrior = newLogPrior;
                oldLogLike = newLL;
                m_message_stream << "accepted 100 percent\n";
                m_iter++; // increase number of iters
            }else if ( std::log(draw) <= logAR ) {
                // probabilistic accept
                if(m_print_to_console)
                    std::cout << "accepting!\n";
                m_iter++; // increase number of iters
                m_ma_accept_rate = 1.0/(m_iter+1.0) + m_iter*m_ma_accept_rate/(m_iter+1.0);
                m_current_theta.takeValues(proposed_theta);
                oldLogPrior = newLogPrior;
                oldLogLike = newLL;
                m_message_stream << "accepted probabilistically\n";
            } else if ( std::log(draw) > logAR ) {
                if(m_print_to_console)
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
            // notice that this is on the untransformed or constrained space
            utils::logParams<numparams>(m_current_theta.getUnTransParams(), m_samples_file_stream);
                
        } // else (not the first iteration)    
    } // while(m_iter < m_num_mcmc_iters) // every iteration
    
    // stop writing thetas and messages
    m_samples_file_stream.close();
    m_message_stream.close();
    
}


#endif //ADA_RWMH_H
