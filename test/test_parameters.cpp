#include <catch2/catch.hpp>
#include <Eigen/Dense>
#include <utility> //move

#include <ssme/parameters.h>

#define NP 4


//class Param_Pack_Fixture
//{
//public:
//
//    Eigen::Matrix<double,NP,1> m_trans_params;
//    Eigen::Matrix<double,NP,1> m_un_trans_params;
//    
//    Param_Pack_Fixture() 
//    {
//        m_trans_params(0) = 1.0; m_un_trans_params(0) = 1.0;
//        m_trans_params(1) = -1.3; m_un_trans_params(1) = std::exp(-1.3); //log
//        m_trans_params(2) = 9.5; m_un_trans_params(2) = std::exp(9.5) / (1.0 + std::exp(9.5)); // logit
//        m_trans_params(3) = .89; m_un_trans_params(3) = (1.0 - std::exp(.89)) / ( -1.0 - std::exp(.89) ); // twice fisher
//    }
//};



using param::trans_type;

TEST_CASE("test constructors", "[trans_type]"){
    
    // 1
    std::vector<trans_type> tts = {trans_type::TT_null, 
                                  trans_type::TT_log, 
                                  trans_type::TT_logit, 
                                  trans_type::TT_twice_fisher};
    Eigen::Matrix<double,NP,1> trans_params;
    trans_params << 1.0, -1.3, 9.5, .89;
    param::pack<double> pp(trans_params, tts);

    // 2
    Eigen::Matrix<double,NP,1> un_trans_params;
    un_trans_params << 1.0, 0.2725318, 0.9999252, .4177803;
    param::pack<double> pp2(un_trans_params, tts, false);

    // 3
    std::vector<std::unique_ptr<param::transform<double>>> derp;
    for(size_t i = 0; i < NP; ++i)
        derp.push_back(std::unique_ptr<param::transform<double> >(new param::twice_fisher_trans<double> ));
    param::pack<double> pp3(trans_params, std::move(derp));
}


TEST_CASE("test assignment", "[trans_type]"){
    
    std::vector<trans_type> tts = {trans_type::TT_null, 
                                  trans_type::TT_log, 
                                  trans_type::TT_logit, 
                                  trans_type::TT_twice_fisher};
    Eigen::Matrix<double,NP,1> trans_params;
    trans_params << 1.0, -1.3, 9.5, .89;    
    param::pack<double> pp1(trans_params, tts);
    Eigen::Matrix<double,NP,1> new_trans_params;
    new_trans_params << 1.0, 1.0, 1.0, 1.0;
    param::pack<double> pp2(new_trans_params, tts);
    pp1.take_values(pp2);
    for(size_t i = 0; i < NP; ++i)
        REQUIRE( std::abs(1.0 - pp1.get_trans_params()(i)) < .00001);
}


TEST_CASE("test transformations", "[trans_type]"){
    
    Eigen::Matrix<double,NP,1> ideal_un_trans_params = {1.0, 0.2725318, 0.9999252, 0.4177803};
    std::vector<trans_type> tts = {trans_type::TT_null, 
                                  trans_type::TT_log, 
                                  trans_type::TT_logit, 
                                  trans_type::TT_twice_fisher};
    Eigen::Matrix<double,NP,1> trans_params;
    trans_params << 1.0, -1.3, 9.5, .89;
    param::pack<double> pp(trans_params, tts);
    for(size_t i = 0; i < NP; ++i)
        REQUIRE( std::abs(ideal_un_trans_params(i) - pp.get_untrans_params()(i)) < .0001);
}

TEST_CASE("test LogJacobians", "[trans_type]"){
    std::vector<trans_type> tts = {trans_type::TT_null, 
                                  trans_type::TT_log, 
                                  trans_type::TT_logit, 
                                  trans_type::TT_twice_fisher};
    Eigen::Matrix<double,NP,1> trans_params;
    trans_params << 1.0, -1.3, 9.5, .89;
    param::pack<double> pp(trans_params, tts);
    REQUIRE( std::abs(-11.6851 - pp.get_log_jacobian()) < .0001); 
    // TODO make this 11.68 calculation more apparent
}


TEST_CASE("test subsetting", "[trans_type]"){
    std::vector<trans_type> tts = {trans_type::TT_null, 
                                  trans_type::TT_log, 
                                  trans_type::TT_logit, 
                                  trans_type::TT_twice_fisher};
    Eigen::Matrix<double,NP,1> trans_params;
    trans_params << 1.0, -1.3, 9.5, .89;
    param::pack<double> pp(trans_params, tts);
    Eigen::Matrix<double,NP,1> ideal_un_trans_params = {1.0, 0.2725318, 0.9999252, 0.4177803};
    for(size_t i = 0; i < NP; ++i)
        REQUIRE( std::abs(trans_params(i) - pp.get_trans_params(i,i)(0)) < .0001);
    for(size_t i = 0; i < NP; ++i)
        REQUIRE( std::abs(ideal_un_trans_params(i) - pp.get_untrans_params(i,i)(0)) < .0001);
}

