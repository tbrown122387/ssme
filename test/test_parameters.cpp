#include <catch2/catch.hpp>
#include <Eigen/Dense>
#include <utility> //move

#include <ssme/parameters.h>

#define NP 4

using param::TransType;

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


TEST_CASE("test constructors", "[TransType]"){
    
    // 1
    std::vector<TransType> tts = {TransType::TT_null, 
                                  TransType::TT_log, 
                                  TransType::TT_logit, 
                                  TransType::TT_twice_fisher};
    Eigen::Matrix<double,NP,1> trans_params;
    trans_params << 1.0, -1.3, 9.5, .89;
    param::paramPack<double> pp(trans_params, tts);

    // 2
    Eigen::Matrix<double,NP,1> un_trans_params;
    un_trans_params << 1.0, 0.2725318, 0.9999252, .4177803;
    param::paramPack<double> pp2(un_trans_params, tts, false);

    // 3
    std::vector<std::unique_ptr<param::paramTransform<double>>> derp;
    for(size_t i = 0; i < NP; ++i)
        derp.push_back(std::unique_ptr<param::paramTransform<double> >(new twiceFisherTrans<double> ));
    param::paramPack<double> pp3(trans_params, std::move(derp));
}


TEST_CASE("test assignment", "[TransType]"){
    
    std::vector<TransType> tts = {TransType::TT_null, 
                                  TransType::TT_log, 
                                  TransType::TT_logit, 
                                  TransType::TT_twice_fisher};
    Eigen::Matrix<double,NP,1> trans_params;
    trans_params << 1.0, -1.3, 9.5, .89;    
    param::paramPack<double> pp1(trans_params, tts);
    Eigen::Matrix<double,NP,1> new_trans_params;
    new_trans_params << 1.0, 1.0, 1.0, 1.0;
    param::paramPack<double> pp2(new_trans_params, tts);
    pp1.takeValues(pp2);
    for(size_t i = 0; i < NP; ++i)
        REQUIRE( std::abs(1.0 - pp1.getTransParams()(i)) < .00001);
}


TEST_CASE("test transformations", "[TransType]"){
    
    Eigen::Matrix<double,NP,1> ideal_un_trans_params = {1.0, 0.2725318, 0.9999252, 0.4177803};
    std::vector<TransType> tts = {TransType::TT_null, 
                                  TransType::TT_log, 
                                  TransType::TT_logit, 
                                  TransType::TT_twice_fisher};
    Eigen::Matrix<double,NP,1> trans_params;
    trans_params << 1.0, -1.3, 9.5, .89;
    param::paramPack<double> pp(trans_params, tts);
    for(size_t i = 0; i < NP; ++i)
        REQUIRE( std::abs(ideal_un_trans_params(i) - pp.getUnTransParams()(i)) < .0001);
}

TEST_CASE("test LogJacobians", "[TransType]"){
    std::vector<TransType> tts = {TransType::TT_null, 
                                  TransType::TT_log, 
                                  TransType::TT_logit, 
                                  TransType::TT_twice_fisher};
    Eigen::Matrix<double,NP,1> trans_params;
    trans_params << 1.0, -1.3, 9.5, .89;
    param::paramPack<double> pp(trans_params, tts);
    REQUIRE( std::abs(-11.6851 - pp.getLogJacobian()) < .0001); 
    // TODO make this 11.68 calculation more apparent
}


TEST_CASE("test subsetting", , "[TransType]"){
    std::vector<TransType> tts = {TransType::TT_null, 
                                  TransType::TT_log, 
                                  TransType::TT_logit, 
                                  TransType::TT_twice_fisher};
    Eigen::Matrix<double,NP,1> trans_params;
    trans_params << 1.0, -1.3, 9.5, .89;
    param::paramPack<double> pp(trans_params, tts);
    Eigen::Matrix<double,NP,1> ideal_un_trans_params = {1.0, 0.2725318, 0.9999252, 0.4177803};
    for(size_t i = 0; i < NP; ++i)
        REQUIRE( std::abs(trans_params(i) - pp.getTransParams(i,i)(0)) < .0001);
    for(size_t i = 0; i < NP; ++i)
        REQUIRE( std::abs(ideal_un_trans_params(i) - pp.getUnTransParams(i,i)(0)) < .0001);
}

