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



TEST_CASE("test transform container", "[transform_container]"){

    // default ctor
    param::transform_container<double> ts;
    REQUIRE( ts.size() == 0);  

    // add atransform and test to make sure it works
    ts.add_transform(param::trans_type::TT_logit);
    REQUIRE( ts.size() == 1);
    REQUIRE( std::abs(ts.get_transforms()[0]->trans(.5)) < .001 ); // logit(.5) == 0
    
    // copy ctor
    param::transform_container<double> ts2 = ts;
    REQUIRE( ts2.size() == 1);
    REQUIRE( std::abs(ts2.get_transforms()[0]->trans(.5)) < .001 ); // logit(.5) == 0

    // assignment operator 
    ts2.add_transform(param::trans_type::TT_log);
    ts = ts2;
    REQUIRE( ts.size() == 2);
    REQUIRE( ts2.size() == 2);
    REQUIRE( std::abs(ts.get_transforms()[1]->trans(1.0)) < .001 ); // log(1) == 0
  
}



TEST_CASE("test constructors", "[pack]"){
    
    // 1
    Eigen::Matrix<double,NP,1> trans_params;
    trans_params << 1.0, -1.3, 9.5, .89;
    param::transform_container<double> ts;
    ts.add_transform(param::trans_type::TT_null);
    ts.add_transform(param::trans_type::TT_log);
    ts.add_transform(param::trans_type::TT_logit);
    ts.add_transform(param::trans_type::TT_twice_fisher);
    param::pack<double> pp(trans_params, ts);

    // 2 copy ctor
    param::pack<double> pp2(pp);

}


TEST_CASE("test assignment", "[pack]"){
 
    param::transform_container<double> ts;
    ts.add_transform(param::trans_type::TT_null);
    ts.add_transform(param::trans_type::TT_log);
    ts.add_transform(param::trans_type::TT_logit);
    ts.add_transform(param::trans_type::TT_twice_fisher);
    Eigen::Matrix<double,NP,1> trans_params;
    trans_params << 1.0, -1.3, 9.5, .89;    
    param::pack<double> pp1(trans_params, ts);

    // test assignment operator
    Eigen::Matrix<double,NP,1> new_trans_params;
    new_trans_params << 1.0, 1.0, 1.0, 1.0;
    param::pack<double> pp2(new_trans_params, ts);
    pp1 = pp2;
    for(size_t i = 0; i < NP; ++i)
        REQUIRE( std::abs(1.0 - pp1.get_trans_params()(i)) < .00001);
}


TEST_CASE("test transformations", "[pack]"){
 
    param::transform_container<double> ts;
    ts.add_transform(param::trans_type::TT_null);
    ts.add_transform(param::trans_type::TT_log);
    ts.add_transform(param::trans_type::TT_logit);
    ts.add_transform(param::trans_type::TT_twice_fisher);   
    Eigen::Matrix<double,NP,1> ideal_un_trans_params = {1.0, 0.2725318, 0.9999252, 0.4177803};
    Eigen::Matrix<double,NP,1> trans_params;
    trans_params << 1.0, -1.3, 9.5, .89;
    param::pack<double> pp(trans_params, ts);
    for(size_t i = 0; i < NP; ++i)
        REQUIRE( std::abs(ideal_un_trans_params(i) - pp.get_untrans_params()(i)) < .0001);
}

TEST_CASE("test LogJacobians", "[pack]"){
    
    param::transform_container<double> ts;
    ts.add_transform(param::trans_type::TT_null);
    ts.add_transform(param::trans_type::TT_log);
    ts.add_transform(param::trans_type::TT_logit);
    ts.add_transform(param::trans_type::TT_twice_fisher);
    Eigen::Matrix<double,NP,1> trans_params;
    trans_params << 1.0, -1.3, 9.5, .89;
    param::pack<double> pp(trans_params, ts);
    REQUIRE( std::abs(-11.6851 - pp.get_log_jacobian()) < .0001); 
    // TODO make this 11.68 calculation more apparent
}


TEST_CASE("test subsetting", "[pack]"){

    param::transform_container<double> ts;
    ts.add_transform(param::trans_type::TT_null);
    ts.add_transform(param::trans_type::TT_log);
    ts.add_transform(param::trans_type::TT_logit);
    ts.add_transform(param::trans_type::TT_twice_fisher);
    Eigen::Matrix<double,NP,1> trans_params;
    trans_params << 1.0, -1.3, 9.5, .89;
    param::pack<double> pp(trans_params, ts);
    Eigen::Matrix<double,NP,1> ideal_un_trans_params = {1.0, 0.2725318, 0.9999252, 0.4177803};
    for(size_t i = 0; i < NP; ++i)
        REQUIRE( std::abs(trans_params(i) - pp.get_trans_params(i,i)(0)) < .0001);
    for(size_t i = 0; i < NP; ++i)
        REQUIRE( std::abs(ideal_un_trans_params(i) - pp.get_untrans_params(i,i)(0)) < .0001);
}

