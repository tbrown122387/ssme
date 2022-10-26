#ifndef PSWARM_FILTER_H
#define PSWARM_FILTER_H

#include <pf/pf_base.h>

#include <functional>
#include <tuple>
#include <type_traits>
#include <utility> // std::pair

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

    /* a model object along with its (parameter-dependent) filtering functions */
    using mod_funcs_pair = std::pair<ModType, std::vector<filt_func>>;

    /* a collection of expectations and a floating point for the log conditional likelihood */
    using mats_and_loglike = std::pair<std::vector<DynMat>, float_type>;

    /* assert that ModType is a proper particle filter model */
    static_assert(std::is_base_of<pf::bases::pf_base<float_type,dimy,dimx>, ModType>::value, 
            "ModType must inherit from a particle filter class.");

private:
  
    /* Models must be instantiated witha  virtual function, so models cannot be instantiated in the Swarm constructor. This flags whether we have done that.*/
    bool m_models_are_not_instantiated;

    /* this is the vector of functions that generates all models' filtering functions*/
    std::vector<state_parm_func> m_proto_funcs;

    /* a collection of models each with a randomly chosen parameter and a vector of functions for each model/parameter */
    std::array<mod_funcs_pair, nparamparts> m_mods_and_funcs;

    /* log p(y_t+1 | y_{1:t}) */
    float_type m_log_cond_like;
    
    /* E[h(x_t)|y_{1:t}] */
    std::vector<DynMat> m_expectations; 

    /* keep track of the number of observations seen in time */
    unsigned int m_num_obs;

    /* thread pool for faster calculations*/
    split_data_thread_pool<osv, mod_funcs_pair, mats_and_loglike, nparamparts> m_tp; 

    /* calls .filter() on a particle filtering object. side effects and the return object are important */
    static mats_and_loglike comp_func(const osv& yt, mod_funcs_pair& pf_funcs){
        pf_funcs.first.filter(yt, pf_funcs.second);
        mats_and_loglike r;
        r.first = pf_funcs.first.getExpectations();
        r.second = pf_funcs.first.getLogCondLike();
        return r; 
    }


    /* inter-thread aggregation mean = inter_thread_agg(intra_thread_aves)) */
    static mats_and_loglike inter_agg_func(const mats_and_loglike& agg, const mats_and_loglike& vec_mats_and_like, unsigned num_threads,
                                           [[maybe_unused]] unsigned num_terms_in_thread){

        // set up required variables 
        mats_and_loglike res = agg;

        // agg log conditional likelihood
        res.second += vec_mats_and_like.second / static_cast<float_t>(num_threads);

        // check if agg has at least one term in it
        // if so, agg is "old"
        bool agg_old = false;
        for(size_t i = 0; i < n_filt_funcs; ++i){
            if(agg.first[i].rows() > 0 || agg.first[i].cols() > 0)
                agg_old = true;
        }

        // aggregate expectation matrices
        // if agg has just been initialized, it is a vector of 0X0 matrices
        // you can't add matrices to each of these 0X0 matrices at the first time point 
        if( agg_old ){
            for(size_t i = 0; i < n_filt_funcs; ++i)
                res.first[i] = res.first[i] + vec_mats_and_like.first[i]/ static_cast<float_t>(num_threads);
        }else{
            for(size_t i = 0; i < n_filt_funcs; ++i)
                res.first[i] = vec_mats_and_like.first[i] / static_cast<float_t>(num_threads);
        }

        //std::cerr << num_threads << ", " << num_terms_in_thread << ", " << vec_mats_and_like.first[0] <<  "\n";
        return res;
    }


    /* agg func = average */
    static mats_and_loglike intra_agg_func(const mats_and_loglike& agg, const mats_and_loglike& vec_mats_and_like, unsigned num_terms_in_thread){

        // set up required variables 
        mats_and_loglike res = agg;

        // agg log conditional likelihood
        res.second += vec_mats_and_like.second / static_cast<float_t>(num_terms_in_thread);
        
        // check if agg has at least one term in it
        // if so, agg is "old"
        bool agg_old = false;
        for(size_t i = 0; i < n_filt_funcs; ++i){
            if(agg.first[i].rows() > 0 || agg.first[i].cols() > 0)
                agg_old = true;
        }

        // aggregate expectation matrices
        // if agg has just been initialized, it is a vector of 0X0 matrices
        // you can't add matrices to each of these 0X0 matrices at the first time point 
        if( agg_old ){
            for(size_t i = 0; i < n_filt_funcs; ++i)
                res.first[i] = res.first[i] + vec_mats_and_like.first[i] / static_cast<float_t>(num_terms_in_thread);
        }else{
            for(size_t i = 0; i < n_filt_funcs; ++i)
                res.first[i] = vec_mats_and_like.first[i] / static_cast<float_t>(num_terms_in_thread);
        }


        //std::cerr << vec_mats_and_like.first[0] / static_cast<float_t>(num_terms_in_thread ) << " intra \n";
        return res;
    }


    /* initalizes empty vector of matrices */
    static mats_and_loglike reset_func(){
        mats_and_loglike vl;
        vl.first.resize(n_filt_funcs);
        vl.second = 0.0;
        return vl;   
    }
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
    explicit Swarm(const std::vector<state_parm_func>& fs, bool parallel = true)
        : m_models_are_not_instantiated(true)
        , m_num_obs(0)
        , m_tp(
                m_mods_and_funcs, 
                &Swarm<ModType,n_filt_funcs,nstateparts,nparamparts,dimy,dimx,dimparam>::comp_func, 
                &Swarm<ModType,n_filt_funcs,nstateparts,nparamparts,dimy,dimx,dimparam>::inter_agg_func, 
                &Swarm<ModType,n_filt_funcs,nstateparts,nparamparts,dimy,dimx,dimparam>::intra_agg_func, 
                &Swarm<ModType,n_filt_funcs,nstateparts,nparamparts,dimy,dimx,dimparam>::reset_func, 
                [](const mats_and_loglike& o){return o;}, 
                parallel) 
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

        // iterate over all parameter values/models
        auto expecs_and_lcl = m_tp.work(yt);
        m_expectations = expecs_and_lcl.first;
        m_log_cond_like = expecs_and_lcl.second;

        // increment number of observations seen
        ++m_num_obs;
    }
  

    /**
     * @brief simulates future observation paths.
     * The index ordering is param,time,particle
     * @param the number of steps into the future you want to simulate observations
     */
    obsSamples simFutureObs(unsigned int num_future_steps) const{
        obsSamples returnMe;
        for(size_t paramSamp = 0; paramSamp < nparamparts; ++paramSamp){
            returnMe[paramSamp] = m_mods_and_funcs[paramSamp].first.sim_future_obs(num_future_steps);
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
            m_mods_and_funcs[i].first = instantiate_mod(untrans_params);
            
            // now create a vector of functions for each model
            std::vector<filt_func> funcs_for_a_mod; 
            for(size_t j = 0; j < n_filt_funcs; ++j){
                funcs_for_a_mod.push_back(gen_filt_func(m_proto_funcs[j], untrans_params));
            } 
            m_mods_and_funcs[i].second = funcs_for_a_mod;
        }

        // set the flag to true so it doesn't have to be done again
        m_models_are_not_instantiated = false;
    }
};



template<typename ModType, size_t n_filt_funcs, size_t nstateparts, size_t nparamparts, size_t dimy, size_t dimx, size_t dimcov, size_t dimparam, bool debug = false>
class SwarmWithCovs {


public:

    /* the floating point number type */
    using float_type      = typename ModType::float_type;

    /* the observation sized vector */
    using osv             = Eigen::Matrix<float_type,dimy,1>;

    /* the observation sized vector */
    using csv             = Eigen::Matrix<float_type,dimcov,1>;

    /* vector to store both observed and covariate. Need this because we need to bundle data for thread pool */
    using ocsv             = Eigen::Matrix<float_type,dimcov+dimy,1>;

    /* the state sized vector */
    using ssv             = Eigen::Matrix<float_type,dimx,1>;

    /** "param sized vector" type alias for linear algebra stuff **/
    using psv             = Eigen::Matrix<float_type, dimparam,1>;

    /* the Matrix type of the underlying model*/
    using DynMat          = typename ModType::dynamic_matrix;

    /* the function that performs filtering on each model */
    using filt_func       = typename ModType::func;

    /* the function that will perform filtering but has a specific parameter */
    using state_cov_parm_func = std::function<const DynMat(const ssv&, const csv&, const psv&)>;

    /* a collection of observation samples, indexed by param,time, then state particle */  // TODO do these need to be stored? or just printed?
    using obsSamples = std::array<std::vector<std::array<osv, nstateparts>>, nparamparts>; 

    /* a model object along with its (parameter-dependent) filtering functions */
    using mod_funcs_pair = std::pair<ModType, std::vector<filt_func>>;

    /* a collection of expectations and a floating point for the log conditional likelihood */
    using mats_and_loglike = std::pair<std::vector<DynMat>, float_type>;

    /* assert that ModType is a proper particle filter model */
    static_assert(std::is_base_of<pf::bases::pf_withcov_base<float_type,dimy,dimx,dimcov>, ModType>::value, 
            "ModType must inherit from the appropriate particle filter class.");

private:
  
    /* Models must be instantiated with a virtual function, so models cannot be instantiated in the Swarm constructor. This flags whether we have done that.*/
    bool m_models_are_not_instantiated;

    /* this is the vector of functions that generates all models' filtering functions*/
    std::vector<state_cov_parm_func> m_proto_funcs;

    /* a collection of models each with a randomly chosen parameter and a vector of functions for each model/parameter */
    std::array<mod_funcs_pair, nparamparts> m_mods_and_funcs;

    /* log p(y_t+1 | y_{1:t}) */
    float_type m_log_cond_like;
    
    /* E[h(x_t)|y_{1:t}] */
    std::vector<DynMat> m_expectations; 

    /* keep track of the number of observations seen in time */
    unsigned int m_num_obs;

    /* thread pool for faster calculations*/
    split_data_thread_pool<ocsv, mod_funcs_pair, mats_and_loglike, nparamparts, debug> m_tp; 

    /* calls .filter() on a particle filtering object. side effects and the return object are important */
    // first element of yt_then_zt is the observation, and then it's the covariate, in that order
    static mats_and_loglike comp_func(const ocsv& yt_then_zt, mod_funcs_pair& pf_funcs){
    	osv yt = yt_then_zt.block(0,0,dimy,1);
    	csv zt = yt_then_zt.block(dimy,0,dimcov,1);
        pf_funcs.first.filter(yt, zt, pf_funcs.second);
        mats_and_loglike r;
        r.first = pf_funcs.first.getExpectations();
        r.second = pf_funcs.first.getLogCondLike();
        return r; 
    }


    /* inter-thread aggregation mean = inter_thread_agg(intra_thread_aves)) */
    static mats_and_loglike inter_agg_func(const mats_and_loglike& agg, const mats_and_loglike& vec_mats_and_like, unsigned num_threads,
                                           [[maybe_unused]] unsigned num_terms_in_thread){

        // set up required variables 
        mats_and_loglike res = agg;

        // agg log conditional likelihood
        res.second += vec_mats_and_like.second / static_cast<float_t>(num_threads);

        // check if agg has at least one term in it
        // if so, agg is "old"
        bool agg_old = false;
        for(size_t i = 0; i < n_filt_funcs; ++i){
            if(agg.first[i].rows() > 0 || agg.first[i].cols() > 0)
                agg_old = true;
        }

        // aggregate expectation matrices
        // if agg has just been initialized, it is a vector of 0X0 matrices
        // you can't add matrices to each of these 0X0 matrices at the first time point 
        if( agg_old ){
            for(size_t i = 0; i < n_filt_funcs; ++i)
                res.first[i] = res.first[i] + vec_mats_and_like.first[i]/ static_cast<float_t>(num_threads);
        }else{
            for(size_t i = 0; i < n_filt_funcs; ++i)
                res.first[i] = vec_mats_and_like.first[i] / static_cast<float_t>(num_threads);
        }

        //std::cerr << num_threads << ", " << num_terms_in_thread << ", " << vec_mats_and_like.first[0] <<  "\n";
        return res;
    }


    /* agg func = average */
    static mats_and_loglike intra_agg_func(const mats_and_loglike& agg, const mats_and_loglike& vec_mats_and_like, unsigned num_terms_in_thread){

        // set up required variables 
        mats_and_loglike res = agg;

        // agg log conditional likelihood
        res.second += vec_mats_and_like.second / static_cast<float_t>(num_terms_in_thread);
        
        // check if agg has at least one term in it
        // if so, agg is "old"
        bool agg_old = false;
        for(size_t i = 0; i < n_filt_funcs; ++i){
            if(agg.first[i].rows() > 0 || agg.first[i].cols() > 0)
                agg_old = true;
        }

        // aggregate expectation matrices
        // if agg has just been initialized, it is a vector of 0X0 matrices
        // you can't add matrices to each of these 0X0 matrices at the first time point 
        if( agg_old ){
            for(size_t i = 0; i < n_filt_funcs; ++i)
                res.first[i] = res.first[i] + vec_mats_and_like.first[i] / static_cast<float_t>(num_terms_in_thread);
        }else{
            for(size_t i = 0; i < n_filt_funcs; ++i)
                res.first[i] = vec_mats_and_like.first[i] / static_cast<float_t>(num_terms_in_thread);
        }


        //std::cerr << vec_mats_and_like.first[0] / static_cast<float_t>(num_terms_in_thread ) << " intra \n";
        return res;
    }


    /* initalizes empty vector of matrices */
    static mats_and_loglike reset_func(){
        mats_and_loglike vl;
        vl.first.resize(n_filt_funcs);
        vl.second = 0.0;
        return vl;   
    }
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
    explicit SwarmWithCovs(const std::vector<state_cov_parm_func>& fs, bool parallel = true)
        : m_models_are_not_instantiated(true)
        , m_num_obs(0)
        , m_tp(
                m_mods_and_funcs,
                &SwarmWithCovs<ModType,n_filt_funcs,nstateparts,nparamparts,dimy,dimx,dimcov,dimparam,debug>::comp_func,
                &SwarmWithCovs<ModType,n_filt_funcs,nstateparts,nparamparts,dimy,dimx,dimcov,dimparam,debug>::inter_agg_func,
                &SwarmWithCovs<ModType,n_filt_funcs,nstateparts,nparamparts,dimy,dimx,dimcov,dimparam,debug>::intra_agg_func,
                &SwarmWithCovs<ModType,n_filt_funcs,nstateparts,nparamparts,dimy,dimx,dimcov,dimparam,debug>::reset_func,
                [](const mats_and_loglike& o){return o;},
                parallel)
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
     * @param the most recent time series observation
     * @param the most recent covariate observation
     */
    void update(const osv& yt, const csv& zt)  {
     
        // instantiate the models if you don't already have them 
        if ( m_models_are_not_instantiated )  finish_construction();

        // TODO: when we average over all parameters/models
        // we are assuming uniform weights because they're being 
        // drawn from the prior...think about generalizing this

        // iterate over all parameter values/models
        ocsv yt_then_zt;
        yt_then_zt.block(0,0,dimy,1) = yt;
        yt_then_zt.block(dimy,0,dimcov,1) = zt;
        auto expecs_and_lcl = m_tp.work(yt_then_zt);
        m_expectations = expecs_and_lcl.first;
        m_log_cond_like = expecs_and_lcl.second;

        // increment number of observations seen
        ++m_num_obs;
    }
  

    /**
     * @brief simulates future observation paths.
     * The index ordering is param,time,particle
     * @param the number of steps into the future you want to simulate observations
     */
    obsSamples simFutureObs(unsigned int num_future_steps) const{
        obsSamples returnMe;
        for(size_t paramSamp = 0; paramSamp < nparamparts; ++paramSamp){
            returnMe[paramSamp] = m_mods_and_funcs[paramSamp].first.sim_future_obs(num_future_steps);
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
    
    /* generate a filter function so that .filter can work on each particle filter  */
    filt_func gen_filt_func(const state_cov_parm_func& in_f, const psv& this_models_params) {
        filt_func out_f = std::bind(in_f, std::placeholders::_1, std::placeholders::_2, this_models_params);
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
            m_mods_and_funcs[i].first = instantiate_mod(untrans_params);
            
            // now create a vector of functions for each model
            std::vector<filt_func> funcs_for_a_mod; 
            for(size_t j = 0; j < n_filt_funcs; ++j){
                funcs_for_a_mod.push_back(gen_filt_func(m_proto_funcs[j], untrans_params));
            } 
            m_mods_and_funcs[i].second = funcs_for_a_mod;
        }

        // set the flag to true so it doesn't have to be done again
        m_models_are_not_instantiated = false;
    }
};


#endif // PSWARM_FILTER_H
