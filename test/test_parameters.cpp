#include <catch2/catch.hpp>
#include <Eigen/Dense>
#include <utility> //move

#include <ssme/parameters.h>


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




//TEST_CASE("test transform container", "[transform_container]"){
//
//    // default ctor
//    param::transform_container<double, 2> ts;
//    REQUIRE( ts.size() == 0);  
//    REQUIRE( ts.capacity() == 2);
//
//    // add a transform and test to make sure it works
//    ts.add_transform(param::trans_type::TT_logit);
//    REQUIRE( ts.size() == 1);
//    REQUIRE( ts.capacity() == 2);
//    REQUIRE( std::abs(ts.get_transforms()[0]->trans(.5)) < .0001 ); // logit(.5) == 0
//    
//    // copy ctor
//    param::transform_container<double, 2> ts2 = ts;
//    REQUIRE( ts2.size() == 1);
//    REQUIRE( ts2.capacity() == 2);
//    REQUIRE( std::abs(ts2.get_transforms()[0]->trans(.5)) < .0001 ); // logit(.5) == 0
//
//    // assignment operator 
//    ts2.add_transform(param::trans_type::TT_log);
//    ts = ts2;
//    REQUIRE( ts.size() == 2);
//    REQUIRE( ts2.size() == 2);
//    REQUIRE( ts.capacity() == 2);
//    REQUIRE( ts2.capacity() == 2);
//    REQUIRE( std::abs(ts.get_transforms()[1]->trans(1.0)) < .001 ); // log(1) == 0
//  
//}



TEST_CASE("test constructors", "[pack]"){
    
    // 1
    // default ctor 
    // size() should be 0, and everything should be a nullptr 
    param::pack<double,4> p1;
//    REQUIRE( p1.size() == 0);
//    REQUIRE( p1.capacity() == 4);
//    REQUIRE_THROWS( p1.get_trans_params() );
//    REQUIRE_THROWS( p1.get_untrans_params());
//    REQUIRE_THROWS( p1.get_log_jacobian() );
//
//    // 2
//    // another constructor that fills everything up
//    Eigen::Matrix<double,4,1> trans_params;
//    trans_params << 1.0, -1.3, 9.5, .89;
//    std::vector<std::string> ts {"null", "log", "logit", "twice_fisher"};
//    param::pack<double,4> p2(trans_params, ts, true);
//    REQUIRE( p2.size() == 4);
//    REQUIRE( p2.capacity() == 4);
////    REQUIRE_THROWS( p1.get_trans_params() );
////    REQUIRE_THROWS( p1.get_untrans_params());
////    REQUIRE_THROWS( p1.get_log_jacobian() );
//
//    // 3 copy ctor
//    param::pack<double,4> p3(p2);
//    param::pack<double,4> p4(p1);
//    REQUIRE( p3.size() == 4);
//    REQUIRE( p4.size() == 0);
//    REQUIRE( p4.capacity() == p3.capacity());


}


TEST_CASE("test assignment", "[pack]"){
 
    std::vector<std::string> ts {"null", "log", "logit", "twice_fisher"};
    Eigen::Matrix<double,4,1> trans_params;
    trans_params << 1.0, -1.3, 9.5, .89;    
    param::pack<double,4> pp1(trans_params, ts);

    // test assignment operator
    Eigen::Matrix<double,4,1> new_trans_params;
    new_trans_params << 1.0, 1.0, 1.0, 1.0;
    param::pack<double,4> pp2(new_trans_params, ts);
    pp1 = pp2;
    for(size_t i = 0; i < 4; ++i)
        REQUIRE( std::abs(1.0 - pp1.get_trans_params()(i)) < .00001);

}


TEST_CASE("test transformations", "[pack]"){
 
    Eigen::Matrix<double,4,1> ideal_un_trans_params = {1.0, 0.2725318, 0.9999252, 0.4177803};
    Eigen::Matrix<double,4,1> trans_params;
    trans_params << 1.0, -1.3, 9.5, .89;
    std::vector<std::string> ts {"null", "log", "logit", "twice_fisher"};
    param::pack<double,4> pp(trans_params, ts);
    for(size_t i = 0; i < 4; ++i)
        REQUIRE( std::abs(ideal_un_trans_params(i) - pp.get_untrans_params()(i)) < .0001);
}

TEST_CASE("test LogJacobians", "[pack]"){
    
    Eigen::Matrix<double,4,1> trans_params;
    trans_params << 1.0, -1.3, 9.5, .89;
    std::vector<std::string> ts {"null", "log", "logit", "twice_fisher"};
    param::pack<double,4> pp(trans_params, ts);
    REQUIRE( std::abs(-11.6851 - pp.get_log_jacobian()) < .0001); 
    // TODO make this 11.68 calculation more apparent
}


TEST_CASE("test subsetting", "[pack]"){

    std::vector<std::string> ts {"null", "log", "logit", "twice_fisher"};
    Eigen::Matrix<double,4,1> trans_params;
    trans_params << 1.0, -1.3, 9.5, .89;
    param::pack<double,4> pp(trans_params, ts);
    Eigen::Matrix<double,4,1> ideal_un_trans_params = {1.0, 0.2725318, 0.9999252, 0.4177803};

    for(size_t i = 0; i < 4; ++i){
//        std::cout << "trans_params(i): " << trans_params(i) 
//                  << "  pp.get_trans_params(i,i)(0): " << pp.get_trans_params(i,i)(0) << "\n";
        REQUIRE( std::abs(trans_params(i) - pp.get_trans_params(i,i)(0)) < .0001);
    }
    for(size_t i = 0; i < 4; ++i)
        REQUIRE( std::abs(ideal_un_trans_params(i) - pp.get_untrans_params(i,i)(0)) < .0001);
}

