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
                         std::placeholders::_2), 
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


// tests if you can run a pool with no additional threads
class MyFixture4 {
public:

    using param_t = std::vector<double>;
    using obs_data_t = std::vector<double>;

    static double d(param_t nums, obs_data_t obs_data) {
        //std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return std::accumulate(nums.begin(), nums.end(), 0.0);
    }

    thread_pool<param_t,obs_data_t,double> pool;
    
    MyFixture4() 
        : pool(d, 1e4, false) { // the only difference between this and above is false!
            pool.add_observed_data( std::vector<double>{999} );
        }
};


TEST_CASE_METHOD(MyFixture4, "test thread pool with single thread", "[thread_pool]")
{

    unsigned num_tries(1e3);
    for(unsigned i = 0; i < num_tries; ++i){
        REQUIRE( std::abs(
                    pool.work(
                        std::vector<double>{1.0, 1.0, 1.0} )
                    - 3.0) < .001  );
    }
}


int comp_func2(double di, int& si){
    int result = si + round(di);
    si++;
    return result;
} 
int agg_func2(int agg, int elem){ return agg + elem; }
int reset_func2(){ return 0; }


TEST_CASE("test new thread pool that preallocates work", "[split_data_thread_pool]")
{

    std::array<int, 100> counters {};
    split_data_thread_pool<double,int,int,100> sdtp(
            counters, 
            comp_func2, 
            agg_func2, 
            reset_func2, 
            [](int o){ return o;}, 
            false);
    int result = sdtp.work(2.1);
    REQUIRE(result == 200);
    for(const auto& elem : counters)
        REQUIRE(elem == 1);
}
