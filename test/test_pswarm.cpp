#include <catch2/catch_all.hpp>
#include <Eigen/Dense>

#include <pf/bootstrap_filter_with_covariates.h>
#include <pf/rv_eval.h>
#include <pf/rv_samp.h>
#include <pf/resamplers.h>

#include <ssme/pswarm_filter.h>
#include <ssme/utils.h> // csv sampler


#define PREC .0001
#define NPARTS 10
#define DIMX 1
#define DIMY 1
#define DIMCOV 1
#define DIMPARAM 4

using FLOATTYPE = double;
using namespace pf;
using namespace pf::filters;
using namespace pf::resamplers;
using pf::bases::GenFutureSimulator;


/**
 * @brief a particle filter class template for a Hull-White stochastic volatility model
 *
 */
template<size_t nparts, typename resampT, typename float_t>
class svol_leverage : public BSFilterWC<nparts, 1, 1, 1, resampT, float_t>
                    , public GenFutureSimulator<1,1,float_t,nparts>
{
public:
    using ssv = Eigen::Matrix<float_t, 1, 1>;
    using osv = Eigen::Matrix<float_t, 1, 1>;
    using cvsv= Eigen::Matrix<float_t,1,1>;

    // parameters
    float_t m_phi;
    float_t m_mu;
    float_t m_sigma;
    float_t m_rho;

    // days to expiration (aka how many days into future you're simulating)
    unsigned int m_dte;

    // use this for sampling
    rvsamp::UnivNormSampler<float_t> m_stdNormSampler; // for sampling

    // ctor
    svol_leverage() = default;
    svol_leverage(const float_t &phi, const float_t &mu, const float_t &sigma, const float_t& rho, unsigned int dte);

    // required by bootstrap filter base class
    float_t logQ1Ev(const ssv &x1, const osv &y1, const cvsv &z1);
    float_t logMuEv(const ssv &x1, const cvsv &z1);
    float_t logGEv(const osv &yt, const ssv &xt, const cvsv& zt);
    auto stateTransSamp(const ssv &xtm1, const cvsv& zt) -> ssv;
    auto q1Samp(const osv &y1, const cvsv& z1) -> ssv;

    // required by FutureSimulator base class
    std::array<ssv,nparts> get_uwtd_samps() const;
    auto gSamp(const ssv &xt) -> osv;
    auto fSamp(const ssv &xtm1, const osv &ytm1) -> ssv;

};


template<size_t nparts, typename resampT, typename float_t>
svol_leverage<nparts,resampT, float_t>::svol_leverage(const float_t &phi, const float_t &mu, const float_t &sigma,
                                                      const float_t &rho, unsigned int dte)
    : m_phi(phi), m_mu(mu), m_sigma(sigma), m_rho(rho), m_dte(dte)
{
}


template<size_t nparts, typename resampT, typename float_t>
auto svol_leverage<nparts, resampT, float_t>::q1Samp(const osv &y1, const cvsv& z1) -> ssv
{
    ssv x1samp;
    x1samp(0) = m_stdNormSampler.sample() * m_sigma / std::sqrt(1.-m_phi*m_phi);
    return x1samp;
}


template<size_t nparts, typename resampT, typename float_t>
auto svol_leverage<nparts, resampT, float_t>::fSamp(const ssv &xtm1, const cvsv& zt) -> ssv
{
    // the covariate zt is ytm1 for this model
    ssv xtsamp;
    float_t mean =  m_mu + m_phi * (xtm1(0) - m_mu) + m_rho*m_sigma*zt(0)*std::exp(-.5*xtm1(0));
    xtsamp(0) = mean + m_stdNormSampler.sample() * m_sigma * std::sqrt( 1.0 - m_phi*m_phi );
    return xtsamp;
}


template<size_t nparts, typename resampT, typename float_t>
float_t svol_leverage<nparts, resampT, float_t>::logGEv(const osv &yt, const ssv &xt, const cvsv& zt)
{
    return rveval::evalUnivNorm<float_t>(
                                    yt(0),
                                    0.0,
                                    std::exp(.5*xt(0)),
                                    true);
}


template<size_t nparts, typename resampT, typename float_t>
auto svol_leverage<nparts, resampT, float_t>::gSamp(const ssv &xt) -> osv {
    osv yt;
    yt(0) = m_stdNormSampler.sample() * std::exp(.5*xt(0));
    return yt;
}


template<size_t nparts, typename resampT, typename float_t>
float_t svol_leverage<nparts, resampT, float_t>::logMuEv(const ssv &x1, const cvsv& z1)
{
    return rveval::evalUnivNorm<float_t>(
                                    x1(0),
                                    0.0,
                                    m_sigma/std::sqrt(1.0 - m_phi*m_phi),
                                    true);
}


template<size_t nparts, typename resampT, typename float_t>
float_t svol_leverage<nparts, resampT, float_t>::logQ1Ev(const ssv &x1samp, const osv &y1, const cvsv& z1)
{
    return rveval::evalUnivNorm<float_t>(x1samp(0), 0.0, m_sigma/std::sqrt(1.0 - m_phi*m_phi), true);
}


template<size_t nparts, typename resampT, typename float_t>
auto svol_leverage<nparts, resampT, float_t>::get_uwtd_samps() const -> std::array<ssv,nparts>
{
    return this->m_particles;
}


/**
 * @brief particle swarm filter (many bootstrap filters)
 * this samples parameters from parameterized distribution
 */
template<size_t n_state_parts, size_t n_param_parts, typename float_t>
class svol_swarm_1 : public SwarmWithCovs<svol_leverage<n_state_parts,mn_resamp_fast1<n_state_parts,DIMX,float_t>, float_t>,
                                1, 
                                n_state_parts, 
                                n_param_parts, 
                                DIMY, DIMX, DIMCOV, DIMPARAM>
{
public:

    using ModType = svol_leverage<n_state_parts,mn_resamp_fast1<n_state_parts,DIMX,float_t>, float_t>;
    using SwarmBase = SwarmWithCovs<ModType, 1, n_state_parts, n_param_parts, DIMY, DIMX, DIMCOV, DIMPARAM>;
    using ssv = Eigen::Matrix<float_t,DIMX,1>;
    using csv = Eigen::Matrix<float_t,DIMCOV,1>;
    using osv = Eigen::Matrix<float_t,DIMY,1>;
    using psv = Eigen::Matrix<float_t, DIMPARAM,1>;
    using state_cov_parm_func = typename SwarmBase::state_cov_parm_func;

private:

    // for sampling from the parameter prior
    rvsamp::UniformSampler<float_t> m_phi_sampler;
    rvsamp::UniformSampler<float_t> m_mu_sampler;
    rvsamp::UniformSampler<float_t> m_sigma_sampler;
    rvsamp::UniformSampler<float_t> m_rho_sampler;

    // days to expiration (aka how many days into future you're simulating)
    unsigned int m_dte;

public:

    svol_swarm_1() = delete;
    
    // default ctor
    svol_swarm_1(float_t phi_l, float_t phi_u, float_t mu_l, float_t mu_u, float_t sig_l, float_t sig_u,
                 float_t rho_l, float_t rho_u, unsigned int dte)
        : SwarmBase() 
    , m_phi_sampler(phi_l, phi_u) 
    , m_mu_sampler(mu_l, mu_u)
    , m_sigma_sampler(sig_l, sig_u)      
    , m_rho_sampler(rho_l, rho_u)
    , m_dte(dte)
    {}

    // ctor
    svol_swarm_1(const std::vector<state_cov_parm_func>& fs, float_t phi_l, float_t phi_u, float_t mu_l, float_t mu_u,
                 float_t sig_l, float_t sig_u, float_t rho_l, float_t rho_u, unsigned int dte)
        : SwarmBase(fs)
        , m_phi_sampler(phi_l, phi_u) 
        , m_mu_sampler(mu_l, mu_u)
        , m_sigma_sampler(sig_l, sig_u)      
        , m_rho_sampler(rho_l, rho_u)
        , m_dte(dte)
    {
    }


    // functions tha twe need to define
    psv samp_untrans_params() override {
        psv untrans_param; //order: phi,mu,sigma,rho
        untrans_param(0) = m_phi_sampler.sample();
        untrans_param(1) = m_mu_sampler.sample();
        untrans_param(2) = m_sigma_sampler.sample();
        untrans_param(3) = m_rho_sampler.sample();
        return untrans_param; 
    }

    ModType instantiate_mod(const psv& untrans_params) {
        // order: phi, beta, sigma 
        return ModType(untrans_params(0), 
                       untrans_params(1),
                       untrans_params(2),
                       untrans_params(3), 
                       m_dte);
    }
};



//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TESTS BELOW HERE 
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

TEST_CASE("test filter with funcs for pswarm filter with covariates instantiated with uniform prior", "[filter method]"){

    // same number of particle sand parameter samples
    using ssv = Eigen::Matrix<FLOATTYPE,DIMX,1>;
    using csv = Eigen::Matrix<FLOATTYPE,DIMCOV,1>;
    using psv = Eigen::Matrix<FLOATTYPE,DIMPARAM,1>;
    using Mat = Eigen::Matrix<FLOATTYPE,Eigen::Dynamic,Eigen::Dynamic>;
    using func = std::function<const Mat(const ssv&, const csv&, const psv&)>;
    std::vector<func> fs;
    auto sillyLambda = [](const ssv& xt, const csv& zt, const psv& pt) -> const Mat { 
        ssv ans; 
        ans(0) = 42.0; 
        return ans; };
    fs.push_back(sillyLambda);
    svol_swarm_1<NPARTS,NPARTS,FLOATTYPE> mod(fs, .8, .99, -.1, .1, .01, .1, -.5, -.01, 10);

    // filter once
    ssv y1;
    csv z1;
    mod.update(y1, z1);

    REQUIRE(std::pow(mod.getLogCondLike(),2) > 0.0);
    REQUIRE(std::abs(mod.getExpectations()[0](0,0) - 42.0) < PREC);
    // check some output TODO more!
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TESTS ABOVE HERE 
//////////////////////////////////////////////////////////////////////////////////////////////////////////////


/**
 * @brief particle swarm filter (many bootstrap filters)
 * this samples parameters from csv file
 */

template<size_t n_state_parts, size_t n_param_parts, typename float_t>
class svol_swarm_2
        : public SwarmWithCovs<svol_leverage<n_state_parts,mn_resamp_fast1<n_state_parts,DIMX,float_t>,float_t>,
                                1, 
                                n_state_parts, 
                                n_param_parts, 
                                DIMY, DIMX, DIMCOV, DIMPARAM>
{
public:

    using ModType = svol_leverage<n_state_parts,mn_resamp_fast1<n_state_parts,DIMX,float_t>, float_t>;
    using SwarmBase = SwarmWithCovs<ModType, 1, n_state_parts, n_param_parts, DIMY, DIMX, DIMCOV, DIMPARAM>;
    using ssv = Eigen::Matrix<float_t,DIMX,1>;
    using csv = Eigen::Matrix<float_t,DIMCOV,1>;
    using osv = Eigen::Matrix<float_t,DIMY,1>;
    using psv = Eigen::Matrix<float_t, DIMPARAM,1>;
    using state_cov_parm_func = typename SwarmBase::state_cov_parm_func;

private:

    // for sampling from mcmc samples
    utils::csv_param_sampler<DIMPARAM, float_t> m_param_sampler;

    // days to expiration (aka how many days into future you're simulating)
    unsigned int m_dte;

public:

    // ctor
    // make sure parameter samples in csv are for phi,mu,sigma,rho
    svol_swarm_2(const std::string &param_csv_filename, const std::vector<state_cov_parm_func>& fs, unsigned int dte)
        : SwarmBase(fs)
    	, m_param_sampler(param_csv_filename)
        , m_dte(dte)
    {
    }


    // functions tha twe need to define
    psv samp_untrans_params() override {
    	return m_param_sampler.samp();
    }

    ModType instantiate_mod(const psv& untrans_params) {
        // order: phi, mu, sigma, rho
        auto param = m_param_sampler.samp();
        return ModType(param(0), 
                       param(1),
                       param(2),
                       param(3), 
                       m_dte);
    }
};


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TESTS BELOW HERE 
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

TEST_CASE("test filter without funcs for pswarm filter with covariates instantiated from csv", "[filter method]"){

    using ssv = Eigen::Matrix<FLOATTYPE,DIMX,1>;
    using csv = Eigen::Matrix<FLOATTYPE,DIMCOV,1>;
    using psv = Eigen::Matrix<FLOATTYPE,DIMPARAM,1>;
    using Mat = Eigen::Matrix<FLOATTYPE,Eigen::Dynamic,Eigen::Dynamic>;
    using func = std::function<const Mat(const ssv&, const csv&, const psv&)>;
    std::vector<func> fs;
    auto sillyLambda = [](const ssv& xt, const csv& zt, const psv& pt) -> const Mat { 
        ssv ans; 
        ans(0) = 42.0; 
        return ans; };
    fs.push_back(sillyLambda);
    svol_swarm_2<NPARTS,NPARTS,FLOATTYPE> mod("test_svol_leverage_samples.csv", fs, 10);

    ssv y1;
    csv z1;
    mod.update(y1,z1);
    REQUIRE(std::pow(mod.getLogCondLike(),2) > 0.0);
    REQUIRE(std::abs(mod.getExpectations()[0](0,0) - 42.0) < PREC);
    std::cout << "expectation...." << mod.getExpectations()[0](0,0) << "\n";
    // check some output TODO more!
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TESTS ABOVE HERE 
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

