#include <catch2/catch.hpp>
#include <ssme/thread_pool.h>

#include <numeric> // accumulate


class MyFixture {
public:

    using param_t = std::vector<double>;
    using obs_data_t = std::vector<double>;

    static double d(param_t nums, obs_data_t obs_data) {
        //std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return std::accumulate(nums.begin(), nums.end(), 0.0);
    }

    thread_pool<param_t,obs_data_t,double> pool;
    
    MyFixture() 
        : pool(d, 1e4, true) {
            pool.add_observed_data( std::vector<double>{999} );
        }
};


TEST_CASE_METHOD(MyFixture, "test thread pool", "[thread_pool]")
{

    unsigned num_tries(1e3);
    for(unsigned i = 0; i < num_tries; ++i){
        REQUIRE( std::abs(
                    pool.work(std::vector<double>{1.0, 1.0, 1.0}) 
                    - 3.0) < .001  
                );
    }
}


class MyFixture2 {
public:
    using param_t = std::vector<double>;
    using obs_data_t = std::vector<double>;

//    static double d(input_t nums) {
//        //std::this_thread::sleep_for(std::chrono::milliseconds(1));
//        return std::accumulate(nums.begin(), nums.end(), 0.0);
//    }

    thread_pool<param_t, obs_data_t, double> pool;
    
    MyFixture2() 
        : pool(
                [](param_t nums, obs_data_t od) -> double{
                    return std::accumulate(nums.begin(), nums.end(), 0.0);
                },
                1e4, 
                true) {
            pool.add_observed_data( std::vector<double>{999} );
        }
};


TEST_CASE_METHOD(MyFixture2, "test thread pool with lambda", "[thread_pool]")
{

    unsigned num_tries(1e3);
    for(unsigned i = 0; i < num_tries; ++i){
        REQUIRE( std::abs(
                    pool.work(
                        std::vector<double>{1.0, 1.0, 1.0} )
                    - 3.0) < .001  );
    }
}


class MyFixture3 {
public:
    using param_t = std::vector<double>;
    using obs_data_t = std::vector<double>;

    double d(param_t theta, obs_data_t ydata) {
        return std::accumulate(theta.begin(), theta.end(), 0.0);
    }

    thread_pool<param_t, obs_data_t, double> pool;
   
    MyFixture3() 
        : pool(std::bind(&MyFixture3::d, 
                         this, 
                         std::placeholders::_1,
                         std::placeholders:: _2), 
               1e4, 
               true) {
            pool.add_observed_data( std::vector<double>{999} );
        }
};


TEST_CASE_METHOD(MyFixture3, "test thread pool with nonstatic method", "[thread_pool]")
{

    unsigned num_tries(1e3);
    for(unsigned i = 0; i < num_tries; ++i){
        REQUIRE( std::abs(
                    pool.work(
                        std::vector<double>{1.0, 1.0, 1.0} )
                    - 3.0) < .001  );
    }
}
