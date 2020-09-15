#include <catch2/catch.hpp>
#include <ssme/thread_pool.h>

#include <numeric> // accumulate


class MyFixture {
public:

    using input_t = std::vector<double>;

    static double d(input_t nums) {
        //std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return std::accumulate(nums.begin(), nums.end(), 0.0);
    }

    thread_pool<input_t,double> pool;
    
    MyFixture() : pool(d, 1e4, true) {}
};


TEST_CASE_METHOD(MyFixture, "test thread pool", "[thread_pool]")
{

    unsigned num_tries(1e3);
    for(unsigned i = 0; i < num_tries; ++i){
        REQUIRE( std::abs(pool.work(std::vector<double>{1.0, 1.0, 1.0})- 3.0) < .001  );
    }
}


class MyFixture2 {
public:

    using input_t = std::vector<double>;

//    static double d(input_t nums) {
//        //std::this_thread::sleep_for(std::chrono::milliseconds(1));
//        return std::accumulate(nums.begin(), nums.end(), 0.0);
//    }

    thread_pool<input_t,double> pool;
    
    MyFixture2() 
        : pool(
                [](input_t nums) -> double{
                    return std::accumulate(nums.begin(), nums.end(), 0.0);
                }
                , 1e4, true) {}
};


TEST_CASE_METHOD(MyFixture2, "test thread pool with lambda", "[thread_pool]")
{

    unsigned num_tries(1e3);
    for(unsigned i = 0; i < num_tries; ++i){
        REQUIRE( std::abs(pool.work(std::vector<double>{1.0, 1.0, 1.0})- 3.0) < .001  );
    }
}
