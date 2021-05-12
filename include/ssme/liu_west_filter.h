#ifndef LIU_WEST_FILTER_H
#define LIU_WEST_FILTER_H

#include <array>
#include <Eigen/Dense>

#include <pf/rv_samp.h>

#include "parameters.h"


/**
 * @class mn_resamp_states_and_params
 * @author taylor
 * @file liu_west_filter.h
 * @brief Class that performs multinomial resampling for the Liu-West filter. 
 * For justification, see page 244 of "Inference in Hidden Markov Models"
 * @tparam nparts the number of particles.
 * @tparam dimx the dimension of each state sample.
 */
template<size_t nparts, size_t dimx, size_t dimparam, typename float_t>
class mn_resamp_states_and_params
{
public:

    /** type alias for linear algebra stuff */
    using ssv = Eigen::Matrix<float_t,dimx,1>;
    
    /** type alias for array of Eigen Matrices */
    using arrayVec = std::array<ssv, nparts>;
    
    /** type alias for array of parameter packs **/
    using arrayParams = std::array<param::pack<float_t,dimparam>, nparts>;
    
    /** type alias for array of float_ts */
    using arrayFloat = std::array<float_t,nparts>;

    
    /**
     * @brief Default constructor. 
     */
    mn_resamp_states_and_params(); // TODO Seed


    /**
     * @brief 
     */
    mn_resamp_states_and_params(unsigned long seed);
    

    /**
     * @brief resamples particles.
     * @param oldStateParts the old state particles
     * @param oldParamParts the old param particles
     * @param oldLogUnNormWts the old log unnormalized weights
     */
    void resampLogWts(arrayVec &oldStateParts, arrayParams &oldParamParts, arrayFloat &oldLogUnNormWts);


private:

    std::mt19937 m_gen; 

    std::uniform_real_distribution<float_t> m_u_sampler;

};


template<size_t nparts, size_t dimx, size_t dimparam, typename float_t>
mn_resamp_states_and_params<nparts,dimx,dimparam,float_t>:: mn_resamp_states_and_params()
    : m_gen{static_cast<std::uint32_t>(
                    std::chrono::high_resolution_clock::now().time_since_epoch().count() )}
    , m_u_sampler(0.0, 1.0)
{
}


template<size_t nparts, size_t dimx, size_t dimparam, typename float_t>
mn_resamp_states_and_params<nparts,dimx,dimparam,float_t>:: mn_resamp_states_and_params(unsigned long seed)
    : m_gen{static_cast<std::uint32_t>(seed)}
    , m_u_sampler(0.0, 1.0)
{
}


template<size_t nparts, size_t dimx, size_t dimparam, typename float_t>
void mn_resamp_states_and_params<nparts,dimx,dimparam,float_t>::resampLogWts(arrayVec &oldStateParts, arrayParams &oldParamParts, arrayFloat &oldLogUnNormWts)
{
    // Using the fancier algorthm detailed on page 244 of IHMM 

    // Create unnormalized weights
    arrayFloat unnorm_weights;
    float_t m = *std::max_element(oldLogUnNormWts.begin(), oldLogUnNormWts.end());
    std::transform(oldLogUnNormWts.begin(), oldLogUnNormWts.end(), unnorm_weights.begin(), 
                    [&m](float_t& d) -> float_t { return std::exp( d - m ); } );
    
    // two things: 
    // 1.) calculate normalizing constant for weights, and 
    // 2.) generate all these exponentials to help with getting order statistics
    // NB: you never need to store E_{N+1}! (this is subtle)
    float_t weight_norm_const(0.0);
    arrayFloat exponentials;
    float_t G(0.0);
    for(size_t i = 0; i < nparts; ++i) {
        weight_norm_const += unnorm_weights[i];
        exponentials[i] = -std::log(m_u_sampler(this->m_gen));   
        G += exponentials[i];
    }
    G -= std::log(m_u_sampler(this->m_gen)); // E_{N+1}

    // see Fig 7.15 in IHMM on page 243
    arrayVec tmpStatePartics = oldStateParts;               
    arrayParams tmpParamPartics = oldParamParts;
    float_t uniform_order_stat(0.0);               // U_{(i)} in the notation of IHMM
    float_t running_sum_normalized_weights(unnorm_weights[0]/weight_norm_const); // \sum_{j=1}^I \omega^j in the notation of IHMM
    float_t one_less_summand(0.0);                 // \sum_{j=1}^{I-1} \omega^j 
    unsigned int idx = 0;
    for(size_t i = 0; i < nparts; ++i){

        uniform_order_stat += exponentials[i]/G; // add a spacing E_i/G
    
        do {
            if( (one_less_summand < uniform_order_stat) && (uniform_order_stat <= running_sum_normalized_weights) ) {
                // select index idx
                tmpStatePartics[i] = oldStateParts[idx];
                tmpParamPartics[i] = oldParamParts[idx];
                break;
            }else{
                // increment idx because it will never be chosen (all the other order statistics are even higher) 
                idx++;
                running_sum_normalized_weights += unnorm_weights[idx]/weight_norm_const;
                one_less_summand += unnorm_weights[idx-1]/weight_norm_const;
            }
        }while(true);
    }

    //overwrite olds with news
    oldStateParts = std::move(tmpStatePartics);
    oldParamParts = tmpParamPartics;
    std::fill(oldLogUnNormWts.begin(), oldLogUnNormWts.end(), 0.0);  
}



//! A base class for the Liu-West Filter.
/**
 * @class LWFilter
 * @author taylor
 * @file liu_west_filter.h
 * @brief Liu West filter 
 * @tparam nparts the number of particles
 * @tparam dimx the dimension of the state's state space
 * @tparam dimy the dimension of each observation 
 * @tparam dimparam the dimension of the parameters
 * @tparam float_t the floating point type
 * @tparam debug whether or not you want to display debug messages
 */
template<size_t nparts, size_t dimx, size_t dimy, size_t dimparam, typename float_t, bool debug=false>
class LWFilter 
{
private:

    /** "state size vector" type alias for linear algebra stuff */
    using ssv = Eigen::Matrix<float_t,dimx,1>;

    /** "observation size vector" type alias for linear algebra stuff */
    using osv = Eigen::Matrix<float_t,dimy,1>;

    /** "param sized vector" type alias for linear algebra stuff **/
    using psv         = Eigen::Matrix<float_t, dimparam,1>;    

    /** "param sized matrix" type alias for linear algebra stuff**/
    using psm	      = Eigen::Matrix<float_t, dimparam,dimparam>;

    /** type alias for linear algebra stuff (dimension of the state ^2) */
    using Mat = Eigen::Matrix<float_t,Eigen::Dynamic,Eigen::Dynamic>;

    /** type alias for linear algebra stuff */
    using arrayStates = std::array<ssv, nparts>;
    
    /** type alias for parameter samples **/
    using arrayParams = std::array<param::pack<float_t,dimparam>, nparts>;
    
    /** type alias for array of float_ts */
    using arrayFloats = std::array<float_t, nparts>;
   
    /** function type that takes in parameter and state and returns dynamically sized matrix  */ 
    using stateParamFunc = std::function<const Mat(const ssv&, const psv&)>;

    /** type alias for array of unsigned ints */
    using arrayUInt = std::array<unsigned int, nparts>;

public:

    /**
     * @brief constructs the Liu-West filter
     * @param transforms that describe how to transform parameters so they are unconstrained
     * @param delta adaptation rate (e.g. .95 or .9 or .99)
     * @param rs the resampling schedule 
     */
    LWFilter(const std::vector<std::string>& transforms, 
              float_t delta,
              const unsigned int &rs=1);
    
    
    /**
     * @brief The (virtual) destructor
     */
    virtual ~LWFilter();
    
     /**
      * @brief Get the latest log conditional likelihood.
      * @return a float_t of the most recent conditional likelihood.
      */
    float_t getLogCondLike () const; 
    
    
    /**
     * @brief return all stored expectations (taken with respect to $p(x_t|y_{1:t})$
     * @return return a std::vector<Mat> of expectations. How many depends on how many callbacks you gave to 
     */
    std::vector<Mat> getExpectations () const;
    

    /**
     * @brief updates filtering distribution on a new datapoint. 
     * Optionally stores expectations of functionals.
     * @param data the most recent data point
     * @param fs a vector of functions if you want to calculate expectations.
     */
    void filter(const osv &data, const std::vector<stateParamFunc>& fs = std::vector<stateParamFunc>());
    

    /**
     * @brief Evaluates the log of mu.
     * @param x1 a Eigen::Matrix<float_t,dimx,1> representing time 1's state.
     * @return a float_t evaluation.
     */
    virtual float_t logMuEv (const ssv &x1, const psv& untrans_p1 ) = 0;
    
    
    /**
     * @brief "Proposes" an untransformed future state to help set up the first stage weights. A good choice is the conditional expectation. 
     * @param xtm1 a Eigen::Matrix<float_t,dimx,1> representing the previous time's state.
     * @param untransformed versions of the current parameter vector 
     * @return a Eigen::Matrix<float_t,dimx,1> representing a likely current time state, to be used by the observation density.
     */
    virtual ssv propMu (const ssv &xtm1, const psv& untrans_p1 ) = 0;
    
    
    /**
     * @brief Samples from q1.
     * @param y1 a Eigen::Matrix<float_t,dimy,1> representing time 1's data point.
     * @return a Eigen::Matrix<float_t,dimx,1> sample for time 1's state.
     */
    virtual ssv q1Samp (const osv &y1, const psv& untrans_p1) = 0;
    
    
    /**
     * @brief Samples from f.
     * @param xtm1 a Eigen::Matrix<float_t,dimx,1> representing the previous time's state.
     * @return a Eigen::Matrix<float_t,dimx,1> state sample for the current time.
     */
    virtual ssv fSamp (const ssv &xtm1, const psv& untrans_p1) = 0;
    
    
    /**
     * @brief Evaluates the log of q1.
     * @param x1 a Eigen::Matrix<float_t,dimx,1> representing time 1's state.
     * @param y1 a Eigen::Matrix<float_t,dimy,1> representing time 1's data observation.
     * @return a float_t evaluation.
     */
    virtual float_t logQ1Ev (const ssv &x1, const osv &y1, const psv& untrans_p1) = 0;
    
    
    /**
     * @brief Evaluates the log of g.
     * @param yt a Eigen::Matrix<float_t,dimy,1> representing time t's data observation.
     * @param xt a Eigen::Matrix<float_t,dimx,1> representing time t's state.
     * @return a float_t evaluation.
     */
    virtual float_t logGEv (const osv &yt, const ssv &xt, const psv& untrans_p1) = 0;

    /**
     * @brief sample a non-transformed parameter vector from the prior
     */
    virtual psv paramPriorSamp() = 0; 

protected:


    /** @brief parameter transforms **/
    std::vector<std::string> m_transforms;

    /** @brief state samples */
    arrayStates m_state_particles;

    /** @brief param samples **/
    arrayParams m_param_particles;    

    /** @brief particle unnormalized weights */
    arrayFloats m_logUnNormWeights;
    
    /** @brief curren time */
    unsigned int m_now; 
    
    /** @brief log p(y_t|y_{1:t-1}) or log p(y1) */
    float_t m_logLastCondLike; 
    
    /** @brief the resampling schedule */
    unsigned int m_rs;
    
    /** @brief resampling object */
    mn_resamp_states_and_params<nparts,dimx,dimparam,float_t> m_resampler;   
 
    /** a multivariate normal sampler for the parameter transitions **/
    pf::rvsamp::MVNSampler<dimparam,float_t> m_mvn_gen;
   
    /** @brief k generator object (default ctor'd)*/
    pf::rvsamp::k_gen<nparts,float_t> m_kGen;
    
    /** @brief expectations E[h(x_t) | y_{1:t}] for user defined "h"s */
    std::vector<Mat> m_expectations;
   
    /** delta, the rate of adjustment (see LW paper for more details)*/
    float_t m_delta; 
};



template<size_t nparts, size_t dimx, size_t dimy, size_t dimparam, typename float_t, bool debug>
LWFilter<nparts, dimx, dimy, dimparam, float_t, debug>::LWFilter(
        const std::vector<std::string>& transforms, 
        float_t delta,
        const unsigned int &rs) 
    : m_transforms(transforms)
    , m_now(0)
    , m_logLastCondLike(0.0)
    , m_rs(rs)
    , m_delta(delta)
{
    std::fill(m_logUnNormWeights.begin(), m_logUnNormWeights.end(), 0.0);
}


template<size_t nparts, size_t dimx, size_t dimy, size_t dimparam, typename float_t, bool debug>
LWFilter<nparts, dimx, dimy, dimparam, float_t, debug>::~LWFilter() { }


template<size_t nparts, size_t dimx, size_t dimy, size_t dimparam, typename float_t, bool debug>
void LWFilter<nparts, dimx, dimy, dimparam, float_t, debug>::filter(const osv &data, const std::vector<stateParamFunc>& fs)
{
    
    if(m_now > 0)
    { 
        
        // set up "first stage weights" to make k index sampler 
        arrayFloats logFirstStageUnNormWeights = m_logUnNormWeights;
        float_t m3(-std::numeric_limits<float_t>::infinity());
        float_t m2(-std::numeric_limits<float_t>::infinity());
        psv old_untrans_param;
        for(size_t ii = 0; ii < nparts; ++ii)  
        {
            // update m3
            if(m_logUnNormWeights[ii] > m3)
                m3 = m_logUnNormWeights[ii];
            
            // sample
	        old_untrans_param 		        = m_param_particles[ii].get_untrans_params();
            logFirstStageUnNormWeights[ii] += logGEv(data, propMu(m_state_particles[ii], old_untrans_param), old_untrans_param); 
            
            // accumulate things
            if(logFirstStageUnNormWeights[ii] > m2)
                m2 = logFirstStageUnNormWeights[ii];
            
            // print stuff if debug mode is on
            if constexpr(debug) {
                std::cout << "time: " << m_now 
                          << ", first stage log unnorm weight: " << logFirstStageUnNormWeights[ii] 
                          << "\n";
            }

        }
               
        // draw ks (indexes) (handles underflow issues)
        arrayUInt myKs = m_kGen.sample(logFirstStageUnNormWeights); 

    	// calculate thetabar and Vt for parameter proposal distribution
        psv thetaBar = psv::Zero();
        psm Vt = psm::Zero();
        psv theta_i;
        for(size_t i = 0; i < nparts; ++i){
            theta_i = m_param_particles[i].get_trans_params();
            thetaBar += theta_i / nparts;
            Vt += (theta_i * theta_i.transpose()) / nparts;
        }
        float_t a { (3.0*m_delta - 1.0)/(2.0*m_delta)};
        float_t hSquared {1.0 - a * a};
    	m_mvn_gen.setCovar( hSquared * Vt);

        // sample parameters and states  
        float_t m1(-std::numeric_limits<float_t>::infinity());
        float_t first_cll_sum(0.0);
        float_t second_cll_sum(0.0);
        float_t third_cll_sum(0.0);
        ssv xtm1k;
        ssv muTk;
        psv mtm1k;
        arrayStates old_state_partics = m_state_particles;
        arrayParams old_param_partics = m_param_particles;
        for(size_t ii = 0; ii < nparts; ++ii)   
        {
            // calclations for log p(y_t|y_{1:t-1}) (using log-sum-exp trick)
            second_cll_sum += std::exp( logFirstStageUnNormWeights[ii] - m2 );
            third_cll_sum  += std::exp( m_logUnNormWeights[ii] - m3 );            
            
            // sample the parameter first, and get ready to sample the rest
            xtm1k = m_state_particles[myKs[ii]];
            mtm1k = a * old_param_partics[myKs[ii]].get_trans_params() + (1.0 - a) * thetaBar;
            param::pack<float_t, dimparam> mtm1k_pack (mtm1k, m_transforms); 
            m_mvn_gen.setMean(mtm1k);
            param::pack<float_t, dimparam> newThetaSamp (m_mvn_gen.sample(), m_transforms);

            // sample parameters, states, adn update unnormalized weights
            m_param_particles[ii]   = newThetaSamp; 
            m_state_particles[ii]   = fSamp(xtm1k, newThetaSamp.get_untrans_params());
            muTk                    = propMu(xtm1k, old_param_partics[myKs[ii]].get_untrans_params());
            m_logUnNormWeights[ii] += logGEv(data, m_state_particles[ii], newThetaSamp.get_untrans_params()) 
                                    - logGEv(data, muTk, mtm1k_pack.get_untrans_params());

            if constexpr(debug) {
                std::cout << "time: " << m_now 
                          << ", transposed state sample: " << m_state_particles[ii].transpose() 
                          << ", transposed nontransformed param sample: " << m_param_particles[ii].get_untrans_param().transpose() 
                          << ", log unnorm weight: " << m_logUnNormWeights[ii] 
                          << "\n";
            }

            // update m1
            if(m_logUnNormWeights[ii] > m1)
                m1 = m_logUnNormWeights[ii];
        }

        // calculate estimate for log of last conditonal likelihood
        for(size_t p = 0; p < nparts; ++p)
             first_cll_sum += std::exp( m_logUnNormWeights[p] - m1 );
        m_logLastCondLike = m1 + std::log(first_cll_sum) + m2 + std::log(second_cll_sum) - 2*m3 - 2*std::log(third_cll_sum);

        if constexpr(debug) 
            std::cout << "time: " << m_now << ", log cond like: " << m_logLastCondLike << "\n";

        // calculate expectations before you resample
        unsigned int fId(0);
        for(auto & h : fs){
    
            Mat testOutput = h(m_state_particles[0], m_param_particles[0].get_untrans_params());
            unsigned int rows = testOutput.rows();
            unsigned int cols = testOutput.cols();
            Mat numer = Mat::Zero(rows,cols);
            float_t denom(0.0);
            
            for(size_t prtcl = 0; prtcl < nparts; ++prtcl){ 
                numer += h(m_state_particles[prtcl], m_param_particles[prtcl].get_untrans_params()) * std::exp(m_logUnNormWeights[prtcl] - m1);
                denom += std::exp(m_logUnNormWeights[prtcl] - m1);
            }
            m_expectations[fId] = numer/denom;

            if constexpr(debug)
                std::cout << "transposed expectation " << fId << "; " << m_expectations[fId] << "\n";

            fId++;
        }

        // if you have to resample
        if( (m_now+1)%m_rs == 0)
            m_resampler.resampLogWts(m_state_particles, m_param_particles, m_logUnNormWeights);

        // advance time
        m_now += 1; 
    
    } else { // (m_now == 0) 

        float_t max(-std::numeric_limits<float_t>::infinity());
        psv untrans_param_samp;
        for(size_t ii = 0; ii < nparts; ++ii)
        {
            // sample param particles
            untrans_param_samp = paramPriorSamp();
            m_param_particles[ii] = param::pack<float_t,dimparam>(untrans_param_samp, m_transforms, false);

            // sample particles
            m_state_particles[ii]  = q1Samp(data, untrans_param_samp);
            m_logUnNormWeights[ii]  = logMuEv(m_state_particles[ii], untrans_param_samp);
            m_logUnNormWeights[ii] += logGEv(data, m_state_particles[ii], untrans_param_samp);
            m_logUnNormWeights[ii] -= logQ1Ev(m_state_particles[ii], data, untrans_param_samp);

            if constexpr(debug) {
                std::cout << "time: " << m_now 
                          << ", transposed state sample: " << m_state_particles[ii].transpose() 
                          << ", transposed nontransformed param sample: " << m_param_particles[ii].get_untrans_param().transpose() 
                          << ", log unnorm weight: " << m_logUnNormWeights[ii] 
                          << "\n";
            }

            // update maximum
            if( m_logUnNormWeights[ii] > max)
                max = m_logUnNormWeights[ii];
        }
        
        // calculate log-likelihood with log-exp-sum trick
        float_t sumExp(0.0);
        for( size_t i = 0; i < nparts; ++i){
            sumExp += std::exp( m_logUnNormWeights[i] - max );
        }
        m_logLastCondLike = - std::log( static_cast<float_t>(nparts) ) + max + std::log(sumExp);
        
        // calculate expectations before you resample
        m_expectations.resize(fs.size());
        unsigned int fId(0);
        for(auto & h : fs){
            
            Mat testOutput = h(m_state_particles[0], m_param_particles[0].get_untrans_params());
            unsigned int rows = testOutput.rows();
            unsigned int cols = testOutput.cols();
            Mat numer = Mat::Zero(rows,cols);
            float_t denom(0.0);
            for(size_t prtcl = 0; prtcl < nparts; ++prtcl){ 
                numer += h(m_state_particles[prtcl], m_param_particles[prtcl].get_untrans_params()) * std::exp(m_logUnNormWeights[prtcl] - max);
                denom += std::exp(m_logUnNormWeights[prtcl] - max);
            }
            m_expectations[fId] = numer/denom;

            if constexpr(debug)
                std::cout << "transposed expectation " << fId << "; " << m_expectations[fId] << "\n";

            fId++;
        }
        
        // resample if you should (automatically normalizes)
        if( (m_now+1) % m_rs == 0)
            m_resampler.resampLogWts(m_state_particles, m_param_particles, m_logUnNormWeights);

        // advance time step
        m_now += 1;    
    }

}


template<size_t nparts, size_t dimx, size_t dimy, size_t dimparam, typename float_t, bool debug>
float_t LWFilter<nparts, dimx, dimy, dimparam, float_t, debug>::getLogCondLike() const
{
    return m_logLastCondLike;
}


template<size_t nparts, size_t dimx, size_t dimy, size_t dimparam, typename float_t, bool debug>
auto LWFilter<nparts, dimx, dimy, dimparam, float_t, debug>::getExpectations() const -> std::vector<Mat>
{
    return m_expectations;
}






//! A base class for a modified version of the Liu-West Filter.
/**
 * @class LWFilter2
 * @author taylor
 * @file liu_west_filter.h
 * @brief Liu West filter but without the auxiliary particle filter thing
 * @tparam nparts the number of particles
 * @tparam dimx the dimension of the state's state space
 * @tparam dimy the dimension of each observation 
 * @tparam dimparam the dimension of the parameters
 * @tparam float_t the floating point type
 * @tparam debug whether or not you want to display debug messages
 */
template<size_t nparts, size_t dimx, size_t dimy, size_t dimparam, typename float_t, bool debug=false>
class LWFilter2
{
public:

    /** "state size vector" type alias for linear algebra stuff */
    using ssv         = Eigen::Matrix<float_t, dimx, 1>; 
    
    /** "obs size vector" type alias for linear algebra stuff */
    using osv         = Eigen::Matrix<float_t, dimy, 1>; // obs size vec
    
    /** "param sized vector" type alias for linear algebra stuff **/
    using psv         = Eigen::Matrix<float_t, dimparam,1>;
    
    /** "param sized matrix" type alias for linear algebra stuff**/
    using psm	      = Eigen::Matrix<float_t, dimparam,dimparam>;
    
    /** type alias for linear algebra stuff */
    using Mat         = Eigen::Matrix<float_t,Eigen::Dynamic,Eigen::Dynamic>;
    
    /** type alias for linear algebra stuff */
    using arrayStates = std::array<ssv, nparts>;
    
    /** type alias for parameter samples **/
    using arrayParams = std::array<param::pack<float_t,dimparam>, nparts>;
    
    /** type alias for array of float_ts */
    using arrayFloats = std::array<float_t, nparts>;
   
    /** function type that takes in parameter and state and returns dynamically sized matrix  */ 
    using stateParamFunc = std::function<const Mat(const ssv&, const psv&)>;

    /**
     * @brief constructs the Liu-West filter
     * @param transforms that describe how to transform parameters so they are unconstrained
     * @param delta adaptation rate (e.g. .95 or .9 or .99)
     * @param rs the resampling schedule 
     */
    LWFilter2(const std::vector<std::string>& transforms, 
              float_t delta,
              const unsigned int &rs=1);
    
   
    /**
     * @brief destructor
     */ 
    ~LWFilter2();
    
   
    /**
     * @brief returns log p(y_t \mid y_{1:t-1})
     */ 
    float_t getLogCondLike() const; 
    
    
    /**
     * @brief return all stored expectations (taken with respect to $p(x_t,theta|y_{1:t})$
     * @return return a std::vector<Mat> of expectations. How many depends on how many callbacks you gave to 
     */
    std::vector<Mat> getExpectations() const;
    
    
    /**
     * @brief updates filtering distribution on a new datapoint. 
     * Optionally stores expectations of functionals.
     * @param data the most recent data point
     * @param fs a vector of functions if you want to calculate expectations.
     */
    void filter(const osv &data, const std::vector<stateParamFunc>& fs = std::vector<stateParamFunc>());
    
    
    /**
     * @brief  Calculate muEv or logmuEv
     * @param x1 is a const Vec& describing the state sample
     * @param untrans_p1 the (not transformed) parameter
     * @return the density or log-density evaluation as a float_t
     */
    virtual float_t logMuEv (const ssv &x1, const psv& untrans_p1) = 0;
    
    
    /**
     * @brief Samples from time 1 proposal 
     * @param y1 is a const Vec& representing the first observed datum
     * @param untrans_p1 is the parameter value sampled from the parameter prior 
     * @return the sample as a Vec
     */
    virtual ssv q1Samp (const osv &y1, const psv& untrans_p1) = 0;    
    
    
    /**
     * @brief Calculate q1Ev or log q1Ev
     * @param x1 is a const Vec& describing the time 1 state sample
     * @param y1 is a const Vec& describing the time 1 datum
     * @param untrans_p1 is the parameter sampled from the parameter prior
     * @return the density or log-density evaluation as a float_t
     */
    virtual float_t logQ1Ev (const ssv &x1, const osv &y1, const psv& untrans_p1 ) = 0;
    
    
    /**
     * @brief Calculate gEv or logGEv
     * @param yt is a const Vec& describing the time t datum
     * @param xt is a const Vec& describing the time t state
     * @param untrans_pt is the parameter 
     * @return the density or log-density evaluation as a float_t
     */
    virtual float_t logGEv (const osv &yt, const ssv &xt, const psv& untrans_pt ) = 0;
    
    
    /**
     * @brief Evaluates the state transition density.
     * @param xt the current state
     * @param xtm1 the previous state
     * @param untrans_pt is the parameter
     * @return a float_t evaluaton of the log density/pmf
     */
    virtual float_t logFEv (const ssv &xt, const ssv &xtm1, const psv& untrans_pt) = 0;
    
    
    /**
     * @brief Samples from the proposal/instrumental/importance density at time t
     * @param xtm1 the previous state sample
     * @param yt the current observation
     * @param untrans_pt is the parameter
     * @return a state sample for the current time xt
     */
    virtual ssv qSamp (const ssv &xtm1, const osv &yt, const psv& untrans_pt) = 0;
    
    
    /**
     * @brief Evaluates the proposal/instrumental/importance density/pmf
     * @param xt current state
     * @param xtm1 previous state
     * @param yt current observation
     * @param untrans_pt is the parameter
     * @return a float_t evaluation of the log density/pmf
     */
    virtual float_t logQEv (const ssv &xt, const ssv &xtm1, const osv &yt, const psv& untrans_pt ) = 0;    
   

    /**
     * @brief sample a non-transformed parameter vector from the prior
     */
    virtual psv paramPriorSamp() = 0; 
 
protected:


    /** @brief parameter transforms **/
    std::vector<std::string> m_transforms;

    /** @brief state samples */
    arrayStates m_state_particles;

    /** @brief param samples **/
    arrayParams m_param_particles;

    /** @brief particle weights */
    arrayFloats m_logUnNormWeights;
    
    /** @brief current time point */
    unsigned int m_now; 

    /** @brief an approximation to log p(y_t \mid y_{1:t-1}) **/
    float_t m_logLastCondLike;  
    
    /** @brief resampling object */
    mn_resamp_states_and_params<nparts,dimx,dimparam,float_t> m_resampler;   
 
    /** a multivariate normal sampler for the parameter transitions **/
    pf::rvsamp::MVNSampler<dimparam,float_t> m_mvn_gen;

    /** @brief expectations E[h(x_t) | y_{1:t}] for user defined "h"s */
    std::vector<Mat> m_expectations; // stores any sample averages the user wants
    
    /** @brief resampling schedule (e.g. resample every __ time points) */
    unsigned int m_resampSched;
   
    /** @brief the discount factor (e.g. .95 or .99) **/
    float_t m_delta; 

    /**
     * @todo implement ESS stuff
     */
  
};


template<size_t nparts, size_t dimx, size_t dimy, size_t dimparam, typename float_t, bool debug>
LWFilter2<nparts,dimx,dimy,dimparam,float_t,debug>::LWFilter2(const std::vector<std::string>& transforms, 
                                                              float_t delta,
                                                              const unsigned int &rs)
                : m_transforms(transforms)
                , m_now(0)
                , m_logLastCondLike(0.0)
                , m_resampSched(rs)
                , m_delta(delta)
{
    std::fill(m_logUnNormWeights.begin(), m_logUnNormWeights.end(), 0.0); // log(1) = 0
}


template<size_t nparts, size_t dimx, size_t dimy, size_t dimparam, typename float_t, bool debug>
LWFilter2<nparts,dimx,dimy,dimparam,float_t,debug>::~LWFilter2() {}

    
template<size_t nparts, size_t dimx, size_t dimy, size_t dimparam, typename float_t, bool debug>
float_t LWFilter2<nparts,dimx,dimy,dimparam,float_t,debug>::getLogCondLike() const
{
    return m_logLastCondLike;
}
    

template<size_t nparts, size_t dimx, size_t dimy, size_t dimparam, typename float_t, bool debug>    
auto LWFilter2<nparts,dimx,dimy,dimparam,float_t,debug>::getExpectations() const -> std::vector<Mat> 
{
    return m_expectations;
}


template<size_t nparts, size_t dimx, size_t dimy, size_t dimparam, typename float_t, bool debug>
void LWFilter2<nparts,dimx,dimy,dimparam,float_t, debug>::filter(const osv &data, const std::vector<stateParamFunc>& fs)
{
    if(m_now > 0)
    {

        // get ready to simulate vector of transformed parameters
    	psv thetaBar = psv::Zero();
    	psm Vt = psm::Zero();	
    	psv theta_i;
    	for(size_t i = 0; i < nparts; ++i){
    	    theta_i = m_param_particles[i].get_trans_params();
    	    thetaBar += theta_i / nparts;
    	    Vt += (theta_i * theta_i.transpose() ) / nparts;
    	}
        float_t a { (3.0*m_delta - 1.0)/(2.0*m_delta)};
        float_t hSquared {1.0 - a * a};
    	m_mvn_gen.setCovar( hSquared * Vt);
    
    	// sample parameters and states 
    	ssv newStateSamp;
        psv new_untrans_param;
    	arrayFloats oldLogUnNormWts = m_logUnNormWeights;
        float_t maxOldLogUnNormWts(-std::numeric_limits<float_t>::infinity());
        for(size_t ii = 0; ii < nparts; ++ii)
        {
            // update max of old logUnNormWts before you change the element
            if (m_logUnNormWeights[ii] > maxOldLogUnNormWts)
                maxOldLogUnNormWts = m_logUnNormWeights[ii];
                
            // sample transformed parameter
    	    m_mvn_gen.setMean(a * m_param_particles[ii].get_trans_params() + (1.0 - a) * thetaBar);
    	    param::pack<float_t, dimparam> newThetaSamp (m_mvn_gen.sample(), m_transforms);
    
    	    // sample state 
     	    new_untrans_param       = newThetaSamp.get_untrans_params();
            newStateSamp            = qSamp(m_state_particles[ii], data, new_untrans_param);
            m_logUnNormWeights[ii] += logFEv(newStateSamp, m_state_particles[ii], new_untrans_param);
            m_logUnNormWeights[ii] += logGEv(data, newStateSamp, new_untrans_param);
            m_logUnNormWeights[ii] -= logQEv(newStateSamp, m_state_particles[ii], data, new_untrans_param);
    
            // overwrite stuff
            m_state_particles[ii] = newStateSamp;
            m_param_particles[ii] = newThetaSamp;

            if constexpr(debug) {
                std::cout << "time: " << m_now 
                          << ", transposed state sample: " << m_state_particles[ii].transpose() 
                          << ", transposed nontransformed param sample: " << m_param_particles[ii].get_untrans_param().transpose() 
                          << ", log unnorm weight: " << m_logUnNormWeights[ii] 
                          << "\n";
            }
        }
       
        // compute estimate of log p(y_t|y_{1:t-1}) with log-exp-sum trick
        float_t maxNumer = *std::max_element(m_logUnNormWeights.begin(), m_logUnNormWeights.end()); //because you added log adjustments
        float_t sumExp1(0.0);
        float_t sumExp2(0.0);
        for(size_t i = 0; i < nparts; ++i){
            sumExp1 += std::exp(m_logUnNormWeights[i] - maxNumer);
            sumExp2 += std::exp(oldLogUnNormWts[i] - maxOldLogUnNormWts);
        }
        m_logLastCondLike = maxNumer + std::log(sumExp1) - maxOldLogUnNormWts - std::log(sumExp2);

        // calculate expectations before you resample
        unsigned int fId(0);
        float_t weightNormConst(0.0);
        for(auto & h : fs){ // iterate over all functions

            Mat testOut = h(m_state_particles[0], m_param_particles[0].get_untrans_params());
            unsigned int rows = testOut.rows();
            unsigned int cols = testOut.cols();
            Mat numer = Mat::Zero(rows,cols);
            float_t denom(0.0);

            for(size_t prtcl = 0; prtcl < nparts; ++prtcl){ // iterate over all particles
                numer += h(m_state_particles[prtcl], m_param_particles[prtcl].get_untrans_params()) * std::exp(m_logUnNormWeights[prtcl] - maxNumer );
                denom += std::exp(m_logUnNormWeights[prtcl] - maxNumer);
            }
            m_expectations[fId] = numer/denom;

            // print stuff if debug mode is on
            if constexpr(debug)
                std::cout << "transposed expectation " << fId << ": " << m_expectations[fId].transpose() << "\n";

            fId++;
        }
 
        // resample if you should
        if( (m_now + 1) % m_resampSched == 0)
            m_resampler.resampLogWts(m_state_particles, m_param_particles, m_logUnNormWeights);

        // advance time
        m_now += 1;       
    } 
    else // (m_now == 0) //time 1
    {
       
        // only need to iterate over particles once
        float_t sumWts(0.0);
        psv untrans_param_samp;
        for(size_t ii = 0; ii < nparts; ++ii)
        {
            // sample param particles
            untrans_param_samp= paramPriorSamp();
            m_param_particles[ii] = param::pack<float_t,dimparam>(untrans_param_samp, m_transforms, false); 

            // sample state particles
            m_state_particles[ii] = q1Samp(data, untrans_param_samp);
            m_logUnNormWeights[ii]  = logMuEv(m_state_particles[ii], untrans_param_samp);
            m_logUnNormWeights[ii] += logGEv(data, m_state_particles[ii], untrans_param_samp);
            m_logUnNormWeights[ii] -= logQ1Ev(m_state_particles[ii], data, untrans_param_samp);

            if constexpr(debug) {
                std::cout << "time: " << m_now 
                          << ", transposed state sample: " << m_state_particles[ii].transpose() 
                          << ", transposed nontransformed param sample: " << m_param_particles[ii].get_untrans_param().transpose() 
                          << ", log unnorm weight: " << m_logUnNormWeights[ii] 
                          << "\n";
            }


        }
       
        // calculate log cond likelihood with log-exp-sum trick
        float_t max = *std::max_element(m_logUnNormWeights.begin(), m_logUnNormWeights.end());
        float_t sumExp(0.0);
        for(size_t i = 0; i < nparts; ++i){
            sumExp += std::exp(m_logUnNormWeights[i] - max);
        }
        m_logLastCondLike = -std::log(nparts) + max + std::log(sumExp);
   
        // calculate expectations before you resample
        m_expectations.resize(fs.size());
        unsigned int fId(0);
        for(auto & h : fs){
            
            Mat testOut = h(m_state_particles[0], m_param_particles[0].get_untrans_params());
            unsigned int rows = testOut.rows();
            unsigned int cols = testOut.cols();
            Mat numer = Mat::Zero(rows,cols);
            float_t denom(0.0);

            for(size_t prtcl = 0; prtcl < nparts; ++prtcl){ 
                numer += h(m_state_particles[prtcl], m_param_particles[prtcl].get_untrans_params()) * std::exp(m_logUnNormWeights[prtcl] - max);
                denom += std::exp(m_logUnNormWeights[prtcl] - max);
            }
            m_expectations[fId] = numer/denom;

            // print stuff if debug mode is on
            if constexpr(debug)
                std::cout << "transposed expectation " << fId << ": " << m_expectations[fId].transpose() << "\n";

            fId++;
        }
   
        // resample if you should
        if( (m_now + 1) % m_resampSched == 0)
            m_resampler.resampLogWts(m_state_particles, m_param_particles, m_logUnNormWeights);
   
        // advance time step
        m_now += 1;   
    }

}

#endif //LIU_WEST_FILTER_H
