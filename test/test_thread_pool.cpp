#include <catch2/catch.hpp>
#include <ssme/thread_pool.h>

#include <numeric> // accumulate


class MyFixture {
public:
    using input_t = std::vector<double>;

    static double d(input_t nums) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return std::accumulate(nums.begin(), nums.end(), 0.0);
    }

    thread_pool<input_t, std::function<double(input_t)>> pool;
    
    MyFixture() : pool(d, 1e4, true) {}
};


TEST_CASE_METHOD(MyFixture, "test thread pool", "[thread_pool]")
{

    unsigned num_tries(10);
    std::vector<std::future<double>> futs;
    for(size_t i = 0; i < num_tries; ++i){
        futs.push_back(pool.work(std::vector<double>{1.0, 1.0, 1.0}));
    }


    for(size_t i = 0; i < num_tries; ++i){
        REQUIRE( std::abs(futs[i].get() - 3.0) < .001  );
    }


}
