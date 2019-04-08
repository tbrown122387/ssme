#ifndef MOD_MSL1_H
#define MOD_MSL1_H

#include <array>
#include <stdexcept> // for runtime_error
#include "rbpf.h"
#include "rv_samp.h"
#include "rv_eval.h"
#include "param_pack.h"


// helper function to convert big parameter container to several containers
template<size_t dimss, size_t dimy>
void unpackMSL1Params(
                    const paramPack& packedParams,
                    Eigen::Matrix<double,dimy,1>& beta, 
                    Eigen::Matrix<double,dimss,1>& phis,
                    Eigen::Matrix<double,dimss,1>& mus,
                    Eigen::Matrix<double,dimss,1>& sigmas,
                    Eigen::Matrix<double,dimy,1>& RSigmasVec,
                    double& lambda,
                    double& p)
{
    // assumes dimss==2
    if(packedParams.getNumParams() != (dimy-1)+(3*dimss) + (dimy) + 2)
        throw std::runtime_error("the parameter dimensions don't line up\n");

    // assumed order: beta, phis, mus, sigmas, RSigmas, lambasPosAndNeg, p    
    // for more information on this logic, see the hand-written notesheet
    beta = Eigen::Matrix<double,dimy,1>::Zero();
    // assuming that dim_factor is 1!
    beta(0,0) = 1.0;
    // recall that htere are dimy*dimx - dimx - (dimx C 2) elements in the beta matrix
    beta.block(1, 0, dimy-1,1) = packedParams.getUnTransParams(0, dimy-2);
    unsigned int G = dimy-1;
    phis = packedParams.getUnTransParams(G, G+dimss-1);
    mus = packedParams.getUnTransParams(G+dimss, G+2*dimss-1);
    sigmas = packedParams.getUnTransParams(G+2*dimss,G+3*dimss-1).array().sqrt().matrix();
    RSigmasVec = packedParams.getUnTransParams(G+3*dimss, G+3*dimss+dimy-1).array().sqrt().matrix();
    lambda = packedParams.getUnTransParams(G+3*dimss+dimy, G+3*dimss+dimy)(0);
    p = packedParams.getUnTransParams(G+4*dimss+dimy-1,G+4*dimss+dimy-1)(0);
}



template<size_t nparts, size_t dimnss, size_t dimss, size_t dimy, typename resampT>
class msl1_rbbpf : public rbpf_hmm_bs<nparts, dimnss, dimss, dimy, resampT>
{
public:

    /** "sampled state size vector" */
    using sssv = Eigen::Matrix<double,dimss,1>;
    /** "not sampled state size vector" */
    using nsssv = Eigen::Matrix<double,dimnss,1>;
    /** "observation size vector" */
    using osv = Eigen::Matrix<double,dimy,1>;
    /** "observation sized matrix" */
    using osm = Eigen::Matrix<double,dimy,dimy>;
    /** "not sampled state size matrix" */
    using nsssMat = Eigen::Matrix<double,dimnss,dimnss>;
    /** " sampled state size matrix" */
    using sssMat = Eigen::Matrix<double,dimss,dimss>; 


    //! alternative constructor
    /**
     * @brief sets everything up.
     * @param K the max number of stocks in a contained panic
     * @param pp the parameters in the form of a paramPack
     * @param calc_d_aves whether or not to calculate dbar and dbbdbar
     */
    msl1_rbbpf(const unsigned int &K, const paramPack& pp, bool calc_d_aves=false);


    //!
    /**
     * @brief returns E[y_{t+1} | y_{1:t}] = \lambda_1 B + \lambda_2 Dbar B
     * @return the vector of size dimy
     */
    osv get_forecast_mean() const;


    //!
    /**
     * @brief returns V[y_{t+1} | x_{2,t}, theta], which is used in the rbpf's averaging scheme inside filter()
     * @return the conditional covariance matrix
     */
    osm get_conditional_var(const sssv& x2t) const;
    

    //! Sample from the first sampler.
    /**
     * @brief samples the second component of the state at time 1.
     * @param y1 most recent datum.
     * @return a Vec sample for x21.
     */
    sssv muSamp();
    
    
    //! Provides the initial mean vector for each HMM filter object.
    /**
     * @brief provides the initial probability vector for each HMM filter object.
     * @param x21 the second state componenent at time 1.
     * @return a Vec representing the probability of each state element.
     */
    nsssv initHMMProbVec(const sssv &x21);
    
    
    //! Provides the transition matrix for each HMM filter object.
    /**
     * @brief provides the transition matrix for each HMM filter object.
     * @param x21 the second state component at time 1. 
     * @return a transition matrix where element (ij) is the probability of transitioning from state i to state j.
     */
    nsssMat initHMMTransMat(const sssv &x21);

    //! Samples the time t second component. 
    /**
     * @brief Samples the time t second component.
     * @param x2tm1 the previous time's second state component.
     * @param yt the current observation.
     * @return a Vec sample of the second state component at the current time.
     */
    sssv fSamp(const sssv &x2tm1);
    
    
    //! How to update your inner HMM filter object at each time.
    /**
     * @brief How to update your inner HMM filter object at each time.
     * @param aModel a HMM filter object describing the conditional closed-form model.
     * @param yt the current time series observation.
     * @param x2t the current second state component.
     */
    void updateHMM(hmm<dimnss,dimy> &aModel, const osv &yt, const sssv &x2t);
    

private:
    rvsamp::MVNSampler<dimss> m_timeOneSampler;
    rvsamp::MVNSampler<dimss> m_transJumpSampler;
    
    osv m_beta_1;   // first factor loading never changes
    osv m_R_sigmas; // observatonal covariance matrix in vector form (std. devs though)
    sssv m_mus;
    sssv m_phis;     // state transition diagonals
    sssv m_sigmas;   // state std devs
    double m_lambda; // factor1 (not state 2) mean
    double m_p; // probability of staying in first state value
    unsigned int m_K;

    std::array<osv,dimnss> m_all_indic_vecs; 
    bool m_store_dbar_aves;
    osm m_dbbdbar;  // shows up in prediction variance expression
};


template<size_t nparts, size_t dimnss, size_t dimss, size_t dimy, typename resampT>
msl1_rbbpf<nparts,dimnss,dimss,dimy,resampT>::msl1_rbbpf(const unsigned int &K, const paramPack& pp, bool calc_d_aves)
    : rbpf_hmm_bs<nparts, dimnss, dimss, dimy, resampT>() // always resamples everytime
    , m_K(K), m_store_dbar_aves(calc_d_aves)
{
    // unpack parameters
    unpackMSL1Params<dimss,dimy>(pp, m_beta_1, m_phis, m_mus, m_sigmas, m_R_sigmas, m_lambda, m_p);

    // set up time one sampler guy
    m_timeOneSampler.setMean(sssv::Zero());
    sssMat sigma0 = ( m_sigmas.array().square() / (1.0 - m_phis.array().square()) ).matrix().asDiagonal();
    m_timeOneSampler.setCovar(sigma0);
    
    // and set up trans jump sampler
    m_transJumpSampler.setMean(sssv::Zero());
    m_transJumpSampler.setCovar(m_sigmas.array().square().matrix().asDiagonal());
    
    // store the entire lexicographic ordering of the 0,1,2..K hot vectors
    unsigned int ctr = 0;
    for(size_t hotNumber = 0; hotNumber < m_K+1; ++hotNumber){ 
            
        // changes each iteration...std::vector because it plays nicely with std::next_permutation
        std::vector<double> dynVec(dimy, 0.0);
        
        // fill in the ones you need and sort
        for(size_t oneIdx = 0; oneIdx < hotNumber; ++oneIdx){ // (0,0,0), (0,0,1), (0,1,1)
            dynVec[oneIdx] = 1.0;
        }
        std::sort(dynVec.begin(), dynVec.end());
        
        //add permutations to allVecs
        do {
            double* ptr = &dynVec[0];
            Eigen::Map<osv> tmp(ptr, dimy); // TODO: do we need dimy?
            m_all_indic_vecs[ctr] = tmp;
            ctr++;
        } while(std::next_permutation(dynVec.begin(), dynVec.end()));
    }

    // store dbar and dbbd_bar
    if(m_store_dbar_aves){
       
       m_dbbdbar = osm::Zero();
       osm D;
       for(size_t x1t = 0; x1t < dimnss; ++x1t){ // for each value of x1t
       
          // we can use the marginal distribution of x_{1,t}'s weights
          // for more info see appendix
          D = m_all_indic_vecs[x1t].asDiagonal();
          m_dbbdbar += D * m_beta_1 * m_beta_1.transpose() * D / (double)dimnss;
       }    
    }
}


template<size_t nparts, size_t dimnss, size_t dimss, size_t dimy, typename resampT>
auto msl1_rbbpf<nparts,dimnss,dimss,dimy,resampT>::get_forecast_mean() const -> osv
{
    if(!m_store_dbar_aves) throw std::runtime_error("you need to be storing dbars to calculate the forecast mean\n");
    return m_lambda*m_beta_1;
}


template<size_t nparts, size_t dimnss, size_t dimss, size_t dimy, typename resampT>
auto msl1_rbbpf<nparts,dimnss,dimss,dimy,resampT>::get_conditional_var(const sssv& x2t) const -> osm
{
    // for a derivation of this expression, see the appendix of the msvol paper
    if(!m_store_dbar_aves) throw std::runtime_error("you need to be storing dbars to calculate these conditional covariances\n");
    osm R = m_R_sigmas.cwiseProduct(m_R_sigmas).asDiagonal();
    return std::exp(m_mus(0) + m_phis(0)*(x2t(0) - m_mus(0)) + .5*m_sigmas(0)*m_sigmas(0))*m_beta_1*m_beta_1.transpose() + 
           std::exp(m_mus(1) + m_phis(1)*(x2t(1) - m_mus(1)) + .5*m_sigmas(1)*m_sigmas(1))*m_dbbdbar + R;
}


template<size_t nparts, size_t dimnss, size_t dimss, size_t dimy, typename resampT>
auto msl1_rbbpf<nparts,dimnss,dimss,dimy,resampT>::muSamp() -> sssv
{
    return m_mus + m_timeOneSampler.sample();
}


template<size_t nparts, size_t dimnss, size_t dimss, size_t dimy, typename resampT>
auto msl1_rbbpf<nparts,dimnss,dimss,dimy,resampT>::initHMMProbVec(const sssv &x21) -> nsssv
{ 
    return nsssv::Constant(1.0/dimnss);
}


template<size_t nparts, size_t dimnss, size_t dimss, size_t dimy, typename resampT>
auto msl1_rbbpf<nparts,dimnss,dimss,dimy,resampT>::initHMMTransMat(const sssv &x21) -> nsssMat
{
    nsssMat I = nsssMat::Identity();
    nsssMat J = nsssMat::Constant(1.0);
    return m_p*I + (J-I)*(1.0-m_p)/(dimnss - 1.0);
}


template<size_t nparts, size_t dimnss, size_t dimss, size_t dimy, typename resampT>
auto msl1_rbbpf<nparts,dimnss,dimss,dimy,resampT>::fSamp(const sssv &x2tm1) -> sssv 
{
    return m_mus + m_phis.asDiagonal() * (x2tm1 - m_mus) + m_transJumpSampler.sample();
}


template<size_t nparts, size_t dimnss, size_t dimss, size_t dimy, typename resampT>
void msl1_rbbpf<nparts,dimnss,dimss,dimy,resampT>::updateHMM(hmm<dimnss,dimy> &aModel, const osv &yt, const sssv &x2t)
{

    // see msl_notes.pdf for more details TODO check 
    // iterate over all possible values of x1t to construct condDensVec
    nsssv condDensVec;
    osm Sigma;    
    osv Dx1B;
    for(size_t i = 0; i < dimnss; ++i){
        Dx1B = m_beta_1.cwiseProduct(m_all_indic_vecs[i]);
        Sigma = m_beta_1 * m_beta_1.transpose() * std::exp(x2t(0))  +  Dx1B * Dx1B.transpose() * std::exp(x2t(1));
        Sigma += m_R_sigmas.array().square().matrix().asDiagonal();
        condDensVec[i] = rveval::evalMultivNorm<dimy>(yt, m_lambda * m_beta_1, Sigma, false);
// this is slower with moderate-dimensional y
//        U.block(0,0,m_dim_obs,1) = m_beta_1;
//        U.block(0,1,m_dim_obs,1) = beta2;
//        condDensVec[c] = densities::evalMultivNormWBDA(yt, 
//                                                        m_lambdas[0] * m_beta_1 + m_lambdas[1] * beta2,
//                                                        m_R_sigmas.array().square().matrix(), 
//                                                        U, 
//                                                        x2t.array().exp().matrix().asDiagonal(), 
//                                                        false); // A+UCU'


    }
    aModel.update(condDensVec);
}


#endif //MOD_MSL1_H
