#ifndef PSWARM_FILTER_H
#define PSWARM_FILTER_H

#include <pf/pf_base.h>

#include <functional>
#include <tuple>
#include <type_traits>

#include "thread_pool.h"


/**
 * @brief object that implements a "particle swarm filter" (see https://arxiv.org/abs/2006.15396)
 * This class will draw parameters from samp_untrans_param (a pure virtual function). This function will
 * usually draw from some distribution that closely approximates an old posterior, and it can even draw
 * from posterior samples in another file.
 * @tparam represents a specific parametric model that has inherited from a pf base class template.
 * @tparam number of functions we want to filter. Each filter func represents the h in E[h(xt)|y_1:t]. 
 */
template<typename ModType, size_t n_filt_funcs, size_t nstateparts, size_t nparamparts, size_t dimy, size_t dimx, size_t dimparam>
class Swarm {


public:

    /* the floating point number type */
    using float_type      = typename ModType::float_type;

    /* the observation sized vector */
    using osv             = Eigen::Matrix<float_type,dimy,1>;

    /* the state sized vector */
    using ssv             = Eigen::Matrix<float_type,dimx,1>;

    /** "param sized vector" type alias for linear algebra stuff **/
    using psv             = Eigen::Matrix<float_type, dimparam,1>;

    /* the Matrix type of the underlying model*/
    using DynMat          = typename ModType::dynamic_matrix;

    /* the function that performs filtering on each model */
    using filt_func       = typename ModType::func;

    /* the function that will perform filtering but has a specific parameter */
    using state_parm_func = std::function<const DynMat(const ssv&, const psv&)>;

    /* a collection of observation samples, indexed by param,time, then state particle */  // TODO do these need to be stored? or just printed?
    using obsSamples = std::array<std::vector<std::array<osv, nstateparts>>, nparamparts>; 


    /* assert that ModType is a proper particle filter model */
    static_assert(std::is_base_of<pf::bases::pf_base<float_type,dimy,dimx>, ModType>::value, 
            "ModType must inherit from a particle filter class.");

private:
  
    /* Models must be instantiated witha  virtual function, so models cannot be instantiated in the Swarm constructor. This flags whether we have done that.*/
    bool m_models_are_not_instantiated;

    /* this is the vector of functions that generates all models' filtering functions*/
    std::vector<state_parm_func> m_proto_funcs;

    /* a collection of models each with a randomly chosen parameter and a vector of functions for each model/parameter */
    std::array<ModType,nparamparts>   m_mods;

    /* a vector of functions for each model (these functions may depend on the model's parameter)  */ //TODO, should we store parameters, or functioins?
    std::array<std::vector<filt_func>, nparamparts> m_funcs;

    /* log p(y_t+1 | y_{1:t}) */
    float_type m_log_cond_like;
    
    /* E[h(x_t)|y_{1:t}] */
    std::vector<DynMat> m_expectations; 

    /* keep track of the number of observations seen in time */
    unsigned int m_num_obs;

    /* thread pool for faster calculations*/
    // TODO
    //    thread_pool<dyn_data_t, static_data_t, float_type> m_pool;


    // TODO how to instantiate empty array of parameter samples
    // TODO: maybe consider storing the entire evidence because it's a sum of products like IS^2 algorithm


public:

    /**
     * @brief ctor
     * TODO: right now there is no way to get the size of each expectation in the vector.
     * Because we are calling the default constructor on each element, they are 0x0 before
     * any data is seen. Perhaps you can back out the dimension earlier to de-complicate thigns
     * Recall that hte particle filter classes' filter() gets passed a vector not an array
     */
    Swarm(const std::vector<state_parm_func>& fs) 
        : m_models_are_not_instantiated(true)
        , m_num_obs(0) 
    { 
        if(fs.size() == n_filt_funcs){
            m_proto_funcs = fs;
            m_expectations.resize(n_filt_funcs);
        }else{
            throw std::invalid_argument("the length of fs needs to agree with the corresponding template parameter");
        }
    } 

    
    /**
     * @brief sample an un-transformed parameter vector from the prior
     */
    virtual psv samp_untrans_params() = 0; 


    /**
     * @brief instantiate a model with an untransformed parameter
     * at the moment, it is up to the user to make sure that
     * the correct ordering of the parameters is used
     */
    virtual ModType instantiate_mod(const psv& untrans_params) = 0; 

    
    /**
     * @brief update the model on a new time point's observation
     * @param the most recent observation
     */
    void update(const osv& yt)  {
     
        // instantiate the models if you don't already have them 
        if ( m_models_are_not_instantiated )  finish_construction();

        // TODO: when we average over all parameters/models
        // we are assuming uniform weights because they're being 
        // drawn from the prior...think about generalizing this

        // zero out stuff that will get re-accumulated across parameter samples
        setLogCondLikeToZero();
        setExpecsToZero();

        // iterate over all parameter values/models
        std::vector<DynMat> tmp_expecs_given_theta;
        float_type Ntheta = static_cast<float_type>(nparamparts);
        for(size_t i = 0; i < nparamparts; ++i) {
            
            // update a model on new data
            // first is the model
            // second is the vector<func>
            m_mods[i].filter(yt, m_funcs[i]);

            // update the conditional likelihood 
            m_log_cond_like += m_mods[i].getLogCondLike();

            // now that we're updated, get the model-specific 
            // filter expectations and then average over all parameters/models
            tmp_expecs_given_theta  = m_mods[i].getExpectations();
            if(m_num_obs > 0 || i > 0) {
                    
                for(size_t j = 0; j < n_filt_funcs; ++j) {
                    m_expectations[j] += tmp_expecs_given_theta[j] / Ntheta;
                }
            } else{ // first time point *and* first parameter

                // m_expectations has length zero at this point
                for(size_t j = 0; j < n_filt_funcs; ++j) { 
                    m_expectations[j] = tmp_expecs_given_theta[j] / Ntheta;
                }
            }
        }
        m_log_cond_like /= static_cast<float_type>(nparamparts);

        // increment number of observations seen
        ++m_num_obs;
    }
  

    /**
     * @brief simulates future observation paths.
     * The index ordering is param,time,particle
     * @param the number of steps into the future you want to simulate observations
     */
    obsSamples simFutureObs(unsigned int num_future_steps){
        obsSamples returnMe;
        for(size_t paramSamp = 0; paramSamp < nparamparts; ++paramSamp){
            returnMe[paramSamp] = m_mods[paramSamp].sim_future_obs(num_future_steps);
        }
        return returnMe; 
    }


    /**
     * @brief get the log of the approx. to the conditional "evidence" log p(y_t+1 | y_0:t, M) 
     * @return the floating point number
     */
    float_type getLogCondLike() const { return m_log_cond_like; }


    /**
     * @brief get the current expectation approx.s E[h(x_t)|y_{1:t}] 
     * @return a vector of Eigen::Mats
     */
    std::vector<DynMat> getExpectations() const { return m_expectations; }


private:
   
    /* set the above to zero so it can be re-accumulated  */
    void setExpecsToZero() {
        for(auto& e : m_expectations) {
            e.setZero();
        }
    }

    /* set log of the above to zero so it can be re-accumulated */
    void setLogCondLikeToZero() { m_log_cond_like = 0.0; }


    /* generate a filter function so that .filter can work on each particle filter  */
    filt_func gen_filt_func(const state_parm_func& in_f, const psv& this_models_params) {
        filt_func out_f = std::bind(in_f, std::placeholders::_1, this_models_params);
        return out_f;
    }


    /* construction must be done with virtual functions. Virtual functions cannot be called from within the constructor. */
    void finish_construction() {

        if( ! m_models_are_not_instantiated ) throw std::runtime_error("you're trying to sample models more than once");

        // instantiate all models and hold onto all model-specific filtering functions
        psv untrans_params;
        for(size_t i = 0; i < nparamparts; ++i){

            // get the parameter vector needed for each model and each model's set of functions
            untrans_params = samp_untrans_params();

            // instantiate a model
            m_mods[i] = instantiate_mod(untrans_params);
            
            // now create a vector of functions for each model
            std::vector<filt_func> funcs_for_a_mod; 
            for(size_t j = 0; j < n_filt_funcs; ++j){
                funcs_for_a_mod.push_back(gen_filt_func(m_proto_funcs[j], untrans_params));
            } 
            m_funcs[i] = funcs_for_a_mod;
        }

        // set the flag to true so it doesn't have to be done again
        m_models_are_not_instantiated = false;
    }
};



#endif // PSWARM_FILTER_H
