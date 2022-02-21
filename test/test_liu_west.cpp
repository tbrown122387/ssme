#include <catch2/catch_all.hpp>
#include <Eigen/Dense>

#include <pf/rv_eval.h>
#include <ssme/liu_west_filter.h>


#define PREC .0001
#define NPARTS 10
#define DIMX 1
#define DIMY 1
#define DIMCOV 1
#define DIMPARAM 4

using FLOATTYPE = double;
using namespace pf;


/**
 * @brief Liu-West filter (sampling parameters from a parameterized
 * parameter distribution with a prng, as opposed to drawing
 * unformly from samples that have been saved to a file)
 * parameter order: phi, mu, sigma, rho
 */
template<size_t nparts, typename float_type>
class svol_lw_1_par
        : public LWFilterWithCovsFutureSimulator<NPARTS, DIMX, DIMY, DIMCOV, DIMPARAM, float_type>
{
public:

    using ssv = Eigen::Matrix<float_type, DIMX, 1>;
    using osv = Eigen::Matrix<float_type, DIMY, 1>;
    using csv = Eigen::Matrix<float_type, DIMCOV,1>;
    using psv = Eigen::Matrix<float_type, DIMPARAM,1>;

private:

    // used for sampling states
    rvsamp::UnivNormSampler<float_type> m_stdNormSampler;

    // for sampling from the parameter prior
    rvsamp::UniformSampler<float_type> m_phi_sampler;
    rvsamp::UniformSampler<float_type> m_mu_sampler;
    rvsamp::UniformSampler<float_type> m_sigma_sampler;
    rvsamp::UniformSampler<float_type> m_rho_sampler;

    // how many days to expiration (aka how many days into the future you are simulating)
    unsigned int m_dte;

public:
    // ctor
    svol_lw_1_par(const float_type& delta, float_type phi_l, float_type phi_u, float_type mu_l, float_type mu_u, float_type sig_l,
                  float_type sig_u, float_type rho_l, float_type rho_u, unsigned dte);

    // pure virtual functions that we need to define
    float_type logMuEv (const ssv &x1, const psv& untrans_p1) override;
    ssv propMu  (const ssv &xtm1, const csv &cov_data, const psv& untrans_old_param) override;
    ssv q1Samp (const osv &y1, const psv& untrans_p1) override;
    ssv fSamp (const ssv &xtm1, const csv &zt, const psv& untrans_new_param) override;
    float_type logQ1Ev (const ssv &x1, const osv &y1, const psv& untrans_p1) override;
    float_type logGEv (const osv &yt, const ssv &xt, const psv& untrans_pt) override;
    psv paramPriorSamp() override;
    osv gSamp(const ssv &xt, const psv &untrans_pt) override;
};


template<size_t nparts, typename float_type>
svol_lw_1_par<nparts,float_type>::svol_lw_1_par(const float_type& delta, float_type phi_l, float_type phi_u, float_type mu_l, float_type mu_u, float_type sig_l, float_type sig_u, float_type rho_l, float_type rho_u, unsigned dte)
    : LWFilterWithCovsFutureSimulator<nparts,DIMX,DIMY,DIMCOV,DIMPARAM,float_type>(
    		std::vector<std::string>{"logit", "null", "log", "twice_fisher"}, // phi, mu, sigma, rho
            	delta)           // PRIORS
    // REMINDER: does output appear to be extremely sensitive to these?
    , m_phi_sampler(phi_l, phi_u)
    , m_mu_sampler(mu_l, mu_u)
    , m_sigma_sampler(sig_l, sig_u)
    , m_rho_sampler(rho_l, rho_u)
    , m_dte(dte)
{
}


template<size_t nparts, typename float_type>
float_type svol_lw_1_par<nparts,float_type>::logMuEv(const ssv &x1, const psv& untrans_p1)
{
    // phi, mu, sigma, rho
    float_type sd = untrans_p1(2) / std::sqrt(1.0 - untrans_p1(0)*untrans_p1(0));
    return rveval::evalUnivNorm<float_type>(x1(0), 0.0, sd, true);
}


template<size_t nparts, typename float_type>
auto svol_lw_1_par<nparts,float_type>::propMu(const ssv &xtm1, const csv &cov_data, const psv& untrans_old_param) -> ssv
{
    // phi, mu, sigma, rho
    ssv xt;
    xt(0) = untrans_old_param(1) + untrans_old_param(0)*(xtm1(0) - untrans_old_param(1));
    xt(0) += cov_data(0)*untrans_old_param(3)*untrans_old_param(2)*std::exp(-.5*xtm1(0));
    return xt;
}


template<size_t nparts, typename float_type>
auto svol_lw_1_par<nparts,float_type>::q1Samp(const osv &y1, const psv& untrans_p1) -> ssv
{
    // phi, mu, sigma, rho
    ssv x1samp;
    x1samp(0) = m_stdNormSampler.sample() * untrans_p1(2) / std::sqrt(1.0 - untrans_p1(0)*untrans_p1(0));
    return x1samp;
}


template<size_t nparts, typename float_type>
auto svol_lw_1_par<nparts,float_type>::fSamp(const ssv &xtm1, const csv &zt, const psv& untrans_new_param) -> ssv
{
    // phi, mu, sigma, rho
    ssv xt;
    xt(0) = untrans_new_param(1) + untrans_new_param(0)*(xtm1(0) - untrans_new_param(1)) + zt(0)*untrans_new_param(3)*untrans_new_param(2)*std::exp(-.5*xtm1(0));
    xt(0) += m_stdNormSampler.sample() * untrans_new_param(2) * std::sqrt( 1.0 - untrans_new_param(3) * untrans_new_param(3));
    return xt;
}


template<size_t nparts, typename float_type>
float_type svol_lw_1_par<nparts,float_type>::logQ1Ev(const ssv &x1, const osv &y1, const psv& untrans_p1)
{
    // phi, mu, sigma, rho
    float_type sd = untrans_p1(2) / std::sqrt(1.0 - untrans_p1(0)*untrans_p1(0));
    return rveval::evalUnivNorm<float_type>(x1(0), 0.0, sd, true);
}


template<size_t nparts, typename float_type>
float_type svol_lw_1_par<nparts,float_type>::logGEv(const osv &yt, const ssv &xt, const psv& untrans_pt)
{
    return rveval::evalUnivNorm<float_type>(yt(0), 0.0, std::exp(.5*xt(0)), true);
}


template<size_t nparts, typename float_type>
auto svol_lw_1_par<nparts,float_type>::paramPriorSamp() -> psv
{
    // phi, mu, sigma, rho
    psv untrans_samp;
    untrans_samp(0) = m_phi_sampler.sample();
    untrans_samp(1) = m_mu_sampler.sample();
    untrans_samp(2) = m_sigma_sampler.sample();
    untrans_samp(3) = m_rho_sampler.sample();
    return untrans_samp;
}

template<size_t nparts, typename float_type>
auto svol_lw_1_par<nparts,float_type>::gSamp(const ssv &xt, const psv &untrans_pt) -> osv
{
    osv ytsamp;
    ytsamp(0) = m_stdNormSampler.sample() * std::exp(.5*xt(0));
    return ytsamp;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TESTS BELOW HERE 
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

TEST_CASE("test filter without funcs for type 1 filters with covariates", "[filter method]"){

    svol_lw_1_par<NPARTS,FLOATTYPE> mod(.99, .8, .99, -.1, .1, .01, .1, -.5, -.01, 10);

    // filter once
    Eigen::Matrix<FLOATTYPE,1,1> y1;
    Eigen::Matrix<FLOATTYPE,1,1> z1;
    mod.filter(y1, z1);

    REQUIRE(std::pow(mod.getLogCondLike(),2) > 0.0);
    // check some output TODO more!
}


TEST_CASE("test filter with funcs for type 1 filters with covariates", "[filter method]"){

    svol_lw_1_par<NPARTS,FLOATTYPE> mod(.99, .8, .99, -.1, .1, .01, .1, -.5, -.01, 10);

    // filter once
    using ssv = Eigen::Matrix<FLOATTYPE,DIMX,1>;
    using csv = Eigen::Matrix<FLOATTYPE,DIMCOV,1>;
    using psv = Eigen::Matrix<FLOATTYPE,DIMPARAM,1>;
    using Mat = Eigen::Matrix<FLOATTYPE,Eigen::Dynamic,Eigen::Dynamic>;
    using func = std::function<const Mat(const ssv&, const csv&, const psv&)>;
    ssv y1;
    csv z1;
    std::vector<func> fs;
    //auto sillyLambda = [](const ssv& xt, const csv& zt, const psv& pt) -> const Mat { return xt; };
    auto sillyLambda = [](const ssv& xt, const csv& zt, const psv& pt) -> const Mat { 
        ssv ans; 
        ans(0) = 42.0; 
        return ans; };
    fs.push_back(sillyLambda);
    mod.filter(y1, z1, fs);

    REQUIRE(std::pow(mod.getLogCondLike(),2) >= 0.0);
    REQUIRE(std::abs(mod.getExpectations()[0](0,0) - 42.0) < PREC);

//    REQUIRE(data.size() == 1);
//    REQUIRE( std::abs(1.23 - data[0](0)) <  PREC);
//    REQUIRE( std::abs(4.56 - data[0](1))< PREC);
    }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TESTS ABOVE HERE 
//////////////////////////////////////////////////////////////////////////////////////////////////////////////


/**
 * @brief "alternative" Liu-West filter
 * parameter order: phi, mu, sigma, rho
 */
template<size_t nparts, typename float_t>
class svol_lw_2_par : public LWFilter2WithCovsFutureSimulator<nparts, DIMX, DIMY, DIMCOV, DIMPARAM, float_t>
{
public:
    using ssv = Eigen::Matrix<float_t, DIMX, 1>;
    using osv = Eigen::Matrix<float_t, DIMY, 1>;
    using csv = Eigen::Matrix<float_t, DIMCOV,1>;
    using psv = Eigen::Matrix<float_t, DIMPARAM,1>;

private:

    // use this for samplign states
    rvsamp::UnivNormSampler<float_t> m_stdNormSampler;  

    // for sampling from the parameter prior
    rvsamp::UniformSampler<float_t> m_phi_sampler;
    rvsamp::UniformSampler<float_t> m_mu_sampler;
    rvsamp::UniformSampler<float_t> m_sigma_sampler;
    rvsamp::UniformSampler<float_t> m_rho_sampler;

    // days to expiration (aka how many days into the future you want to simulate)
    unsigned int m_dte;

public:
    // ctor
    svol_lw_2_par() = delete;
    svol_lw_2_par(const float_t &delta, float_t phi_l, float_t phi_u, float_t mu_l, float_t mu_u, float_t sig_l, float_t sig_u, float_t rho_l, float_t rho_u, unsigned m_dte);

    // functions tha twe need to define
    float_t logMuEv (const ssv &x1, const psv& untrans_p1) override;
    ssv q1Samp (const osv &y1, const psv& untrans_p1) override;
    float_t logQ1Ev (const ssv &x1, const osv &y1, const psv& untrans_p1) override;
    float_t logGEv (const osv &yt, const ssv &xt, const psv& untrans_pt) override;
    float_t logFEv (const ssv &xt, const ssv &xtm1, const csv &cov_data, const psv& untrans_pt) override;
    ssv qSamp (const ssv &xtm1, const osv &yt, const csv &cov_data, const psv& untrans_pt) override;
    float_t logQEv (const ssv &xt, const ssv &xtm1, const osv &yt, const csv &cov_data, const psv& untrans_pt) override;
    psv paramPriorSamp() override;
    osv gSamp(const ssv &xt, const psv &untrans_pt) override;

};


template<size_t nparts, typename float_t>
svol_lw_2_par<nparts,float_t>::svol_lw_2_par(const float_t& delta, float_t phi_l, float_t phi_u, float_t mu_l,
                                             float_t mu_u, float_t sig_l, float_t sig_u, float_t rho_l, float_t rho_u,
                                             unsigned dte)
    : LWFilter2WithCovsFutureSimulator<nparts,DIMX,DIMY,DIMCOV,DIMPARAM,float_t>(
    				std::vector<std::string> {"logit", "null", "log", "twice_fisher"}, // phi, mu, sigma, rho
            			delta)           // PRIORS    // REMINDER: does output appear to be extremely sensitive to these?
    // REMINDER: does output appear to be extremely sensitive to these?
    , m_phi_sampler(phi_l, phi_u) 
    , m_mu_sampler(mu_l, mu_u)
    , m_sigma_sampler(sig_l, sig_u)      
    , m_rho_sampler(rho_l, rho_u)
    , m_dte(dte)
{
}


template<size_t nparts, typename float_t>
float_t svol_lw_2_par<nparts,float_t>::logMuEv(const ssv &x1, const psv& untrans_p1)
{
    // phi, mu, sigma, rho
    float_t sd = untrans_p1(2) / std::sqrt(1.0 - untrans_p1(0)*untrans_p1(0));
    return rveval::evalUnivNorm<float_t>(x1(0), 0.0, sd, true);
}


template<size_t nparts, typename float_t>
auto svol_lw_2_par<nparts,float_t>::q1Samp(const osv &y1, const psv& untrans_p1) -> ssv
{
    // phi, mu, sigma, rho
    ssv x1samp;
    x1samp(0) = m_stdNormSampler.sample() * untrans_p1(2) / std::sqrt(1.0 - untrans_p1(0)*untrans_p1(0));
    return x1samp;
}


template<size_t nparts, typename float_t>
float_t svol_lw_2_par<nparts,float_t>::logQ1Ev(const ssv &x1, const osv &y1, const psv& untrans_p1)
{
    // phi, mu, sigma, rho
    float_t sd = untrans_p1(2) / std::sqrt(1.0 - untrans_p1(0)*untrans_p1(0));
    return rveval::evalUnivNorm<float_t>(x1(0), 0.0, sd, true);
}


template<size_t nparts, typename float_t>
float_t svol_lw_2_par<nparts,float_t>::logGEv(const osv &yt, const ssv &xt, const psv& untrans_pt)
{
    return rveval::evalUnivNorm<float_t>(yt(0), 0.0, std::exp(.5*xt(0)), true);
}


template<size_t nparts, typename float_t>
float_t svol_lw_2_par<nparts,float_t>::logFEv(const ssv &xt, const ssv &xtm1, const csv &cov_data, const psv& untrans_pt)
{
    // phi, mu, sigma, rho
    float_t mean = untrans_pt(1) + untrans_pt(0)*(xtm1(0) - untrans_pt(1)) + cov_data(0)*untrans_pt(3)*untrans_pt(2)*std::exp(-.5*xtm1(0));
    float_t sd = untrans_pt(2) * std::sqrt( 1.0 - untrans_pt(3) * untrans_pt(3));
    return rveval::evalUnivNorm<float_t>(xt(0), mean, sd, true);
}


template<size_t nparts, typename float_t>
auto svol_lw_2_par<nparts,float_t>::qSamp(const ssv &xtm1, const osv &yt, const csv &cov_data, const psv& untrans_pt) -> ssv
{
    // phi, mu, sigma, rho
    ssv xt;
    float_t mean = untrans_pt(1) + untrans_pt(0)*(xtm1(0) - untrans_pt(1)) + cov_data(0)*untrans_pt(3)*untrans_pt(2)*std::exp(-.5*xtm1(0));
    xt(0) = mean + m_stdNormSampler.sample() * untrans_pt(2) * std::sqrt( 1.0 - untrans_pt(3) * untrans_pt(3));
    return xt;
}


template<size_t nparts, typename float_t>
float_t svol_lw_2_par<nparts,float_t>::logQEv(const ssv &xt, const ssv &xtm1, const osv &yt, const csv &cov_data, const psv& untrans_pt)
{
    // phi, mu, sigma, rho
    float_t mean = untrans_pt(1) + untrans_pt(0)*(xtm1(0) - untrans_pt(1)) + cov_data(0)*untrans_pt(3)*untrans_pt(2)*std::exp(-.5*xtm1(0));
    float_t sd = untrans_pt(2) * std::sqrt( 1.0 - untrans_pt(3) * untrans_pt(3));
    return rveval::evalUnivNorm<float_t>(xt(0), mean, sd, true);
}


template<size_t nparts, typename float_t>
auto svol_lw_2_par<nparts,float_t>::paramPriorSamp() -> psv
{
    // phi, mu, sigma, rho
    psv untrans_samp;
    untrans_samp(0) = m_phi_sampler.sample();
    untrans_samp(1) = m_mu_sampler.sample();
    untrans_samp(2) = m_sigma_sampler.sample();
    untrans_samp(3) = m_rho_sampler.sample();
    return untrans_samp;
}

template<size_t nparts, typename float_t>
auto svol_lw_2_par<nparts,float_t>::gSamp(const ssv &xt, const psv &untrans_pt) -> osv
{
    osv ytsamp;
    ytsamp(0) = m_stdNormSampler.sample() * std::exp(.5*xt(0));
    return ytsamp;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TESTS BELOW HERE 
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

TEST_CASE("test filter without funcs for type 2 filters with covariates", "[filter method]"){

    svol_lw_2_par<NPARTS,FLOATTYPE> mod(.99, .8, .99, -.1, .1, .01, .1, -.5, -.01, 10);

    // filter once
    Eigen::Matrix<FLOATTYPE,1,1> y1;
    Eigen::Matrix<FLOATTYPE,1,1> z1;
    mod.filter(y1, z1);

    REQUIRE(std::pow(mod.getLogCondLike(),2) > 0.0);
    // check some output TODO more!
}


TEST_CASE("test filter with funcs for type 2 filters with covariates", "[filter method]"){

    svol_lw_2_par<NPARTS,FLOATTYPE> mod(.99, .8, .99, -.1, .1, .01, .1, -.5, -.01, 10);

    // filter once
    using ssv = Eigen::Matrix<FLOATTYPE,DIMX,1>;
    using csv = Eigen::Matrix<FLOATTYPE,DIMCOV,1>;
    using psv = Eigen::Matrix<FLOATTYPE,DIMPARAM,1>;
    using Mat = Eigen::Matrix<FLOATTYPE,Eigen::Dynamic,Eigen::Dynamic>;
    using func = std::function<const Mat(const ssv&, const csv&, const psv&)>;
    ssv y1;
    csv z1;
    std::vector<func> fs;
    //auto sillyLambda = [](const ssv& xt, const csv& zt, const psv& pt) -> const Mat { return xt; };
    auto sillyLambda = [](const ssv& xt, const csv& zt, const psv& pt) -> const Mat { 
        ssv ans; 
        ans(0) = 42.0; 
        return ans; };
    fs.push_back(sillyLambda);
    mod.filter(y1, z1, fs);

    REQUIRE(std::pow(mod.getLogCondLike(),2) >= 0.0);
    REQUIRE(std::abs(mod.getExpectations()[0](0,0) - 42.0) < PREC);

//    REQUIRE(data.size() == 1);
//    REQUIRE( std::abs(1.23 - data[0](0)) <  PREC);
//    REQUIRE( std::abs(4.56 - data[0](1))< PREC);
    }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// TESTS ABOVE HERE 
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

