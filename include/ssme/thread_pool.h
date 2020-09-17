#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <future>


/**
 * @class join_threads
 * @brief RAII thread killer
 */
class join_threads
{
    std::vector<std::thread>& m_threads;
public:

    explicit join_threads(std::vector<std::thread>& threads_)
        : m_threads(threads_) {}

    ~join_threads() {
        for(unsigned long i=0; i < m_threads.size(); ++i) {
            if(m_threads[i].joinable())
                m_threads[i].join();
        }
    }
};


/**
 * @brief thread_pool
 * single param writer, many concurrent parameter readers
 * output is averaged in a thread-safe way
 */
template<typename dyn_data_t, typename static_data_t, typename func_output_t >
class thread_pool
{
public:   
    using F = std::function<func_output_t(dyn_data_t, static_data_t)>; 
private:
    static_assert( std::is_floating_point<func_output_t>::value,  
                   "function output type must be floating point");

    mutable std::mutex m_ac_mut; // locks the average and count of comps
    func_output_t m_working_ave;
    std::atomic<unsigned> m_count;
    unsigned m_total_calcs;
    std::promise<func_output_t> m_out;

    std::atomic_bool m_done;
    std::atomic_bool m_has_an_input;
    std::vector<std::thread> m_threads;
    join_threads m_joiner;

    F m_f; // same function always used
    dyn_data_t m_param; // changed occasionally by a single writer
    static_data_t m_observed_data; // data you're conditioning on that never changes  
    mutable std::shared_mutex m_param_mut; // shared lock for reading input
   

    /**
     * @brief function running on all threads
     */ 
    void worker_thread() {

        while(!m_done)
        {
            if(m_has_an_input){
               
                // call the expensive function
                std::shared_lock<std::shared_mutex> param_lock(m_param_mut);
                func_output_t val = m_f(m_param, m_observed_data);

                // write it to the average and increment count
                // do this in thread-safe way with mutex
                std::lock_guard<std::mutex> out_lock{m_ac_mut};
                if( m_count.load() < m_total_calcs ) {
                    m_working_ave += val / m_total_calcs;
                    m_count++;
                }else if( m_count.load() == m_total_calcs){
                    m_out.set_value(m_working_ave);
                    m_has_an_input = false;
                    m_count++;
                }

            }else{
                  std::this_thread::yield();
            }
        }
    }

public:
   
    /**
     * @brief ctor spawns working threads
     * @param f the function that gets called a bunch of times
     * @param num_comps the number of times to call f
     * @param mt do you want multiple threads 
     */  
    thread_pool(F f, const static_data_t& obs_data, unsigned num_comps, bool mt = true) 
        : m_working_ave(0.0)
        , m_count(0)
        , m_total_calcs(num_comps)
        , m_done(false)
        , m_has_an_input(false)
        , m_joiner(m_threads)
        , m_observed_data(obs_data)
        , m_f(f) 
    {

        unsigned nt = std::thread::hardware_concurrency(); // should I subtract one?
        unsigned const thread_count = ((nt > 1) && mt) ? nt : 1;

        try {
            for(unsigned i=0; i< thread_count; ++i) {
                m_threads.push_back( std::thread(&thread_pool::worker_thread, this));
            }
        } catch(...) {
            m_done=true;
            throw;
        }
    }


    /**
     * @brief destructor
     */
    ~thread_pool() {
        m_done=true;
    }


    /**
     * @brief changes the shared data member, 
     * resets the num_comps_left variable, 
     * resets the accumulator thing to 0, and
     * resets the promise object
     */
    func_output_t work(dyn_data_t new_param) {
        
        {
            std::unique_lock<std::shared_mutex> param_lk(m_param_mut);
            m_param = new_param;
        }

        {
            std::unique_lock<std::mutex> ave_lock(m_ac_mut); 
            m_count = 0;
            m_working_ave = 0.0;
        }

        m_out = std::promise<func_output_t>();
        m_has_an_input = true; 
        return m_out.get_future().get();
    }
};



#endif // THREAD_POOL_H
