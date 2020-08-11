#include <catch2/catch.hpp>
#include <Eigen/Dense>

#include <ssme/utils.h>


#define PREC .0001

TEST_CASE("data_reader_test", "[readInData]"){
    // cannot have header!
    // need to know num cols
    std::vector<Eigen::Matrix<double,2,1>> data = utils::readInData<2,double>("test_data.csv");
    REQUIRE(data.size() == 1);
    REQUIRE( std::abs(1.23 - data[0](0)) <  PREC);
    REQUIRE( std::abs(4.56 - data[0](1))< PREC);
}

