#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <future>
#include <map>
#include <array>


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
 * @brief Here we have many concurrent parameter readers, but only a single parameter writer.
 * This thread pool owns one function that returns one (random) floating point number. The pool sits ready to perform calculations
 * on any new parameter value. Once a new parameter value is received, this pool calls its function a fixed number of times, 
 * and all of the function output is averaged in a thread-safe way. For our particular applications, this function will also depend 
 * on observed data that doesn't change once the thread pool has been initialized.
 * @tparam dyn_data_t dynamic data type. The type of input that gets changed repeatedly.
 * @tparam static_data_t the type of input that only gets set once
 * @tparam the type of function output 
 */
template<typename dyn_data_t, typename static_data_t, typename func_output_t >
class thread_pool
{
public:  

    /* type alias for the type of function this thread pool owns */ 
    using F = std::function<func_output_t(dyn_data_t, static_data_t)>; 

private:
    static_assert( std::is_floating_point<func_output_t>::value,  
                   "function output type must be floating point");


    /* shared lock for reading in new parameter values */  
    mutable std::shared_mutex m_param_mut; 

    /* locks the average and count of comps */
    mutable std::mutex m_ac_mut; 
 
    /* the number of calls that have been completed so far */
    std::atomic<unsigned> m_count;

    /* flag for if pool is operational */
    std::atomic_bool m_done;

    /* flag for if there is a new input */
    std::atomic_bool m_has_an_input;
    
    /* the accumulated variable (working average) */
    func_output_t m_working_ave;

    /* the unchanging desired number of function calls you want for each fresh new input */
    unsigned m_total_calcs;

    /* promised output of an average */
    std::promise<func_output_t> m_out;

    /* our function that gets called every time */
    F m_f; 

    /* the dynamic parameter vector that is used as an input to the function, which occasionally gets changed by a single writer  */
    dyn_data_t m_param;

    /* the unchanging observed data that is used as an input to the function */ 
    static_data_t m_observed_data; 
 
    /* whether the observed "static" data has been added to the thread pool yet */ 
    bool m_no_data_yet; 
   
    /* the raw threads */
    std::vector<std::thread> m_threads;
    
    /* the RAII thread killer (defined above) */
    join_threads m_joiner;


    /**
     * @brief This function runs on all threads, and continuously waits for work to do. When a new parameter comes, 
     * calculations begin to be performed, and all their outputs are averaged together.
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
     * @brief The ctor spawns the working threads and gets ready to start doing work.
     * @param f the function that gets called a bunch of times.
     * @param num_comps the number of times to call f with each new parameter input.
     * @param mt do you want multiple threads 
     */  
    thread_pool(F f, unsigned num_comps, bool mt = true) 
        : m_count(0)
        , m_done(false)
        , m_has_an_input(false)
	    , m_working_ave(0.0)
        , m_total_calcs(num_comps)
        , m_f(f)
        , m_no_data_yet(true) 
        , m_joiner(m_threads)

    {

        unsigned nt = std::thread::hardware_concurrency(); 
        unsigned thread_count = ((nt > 1) && mt) ? nt : 1 ;
        if ( thread_count > 1) thread_count -= 1;

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
     * @brief add observed data once before any calculations are performed
     * @param obs_data the entire collection of observed data
     */
    void add_observed_data(const static_data_t& obs_data)
    {
        m_observed_data = obs_data;
        m_no_data_yet = false;
    }


    /**
     * @brief destructor
     */
    ~thread_pool() {
        m_done=true;
    }


    /**
     * @brief changes the shared data member, then resets some variables, then starts all the work and returns the average.
     * @param the new parameter input 
     * @return a floating point average 
     */
    func_output_t work(dyn_data_t new_param) {

        if( m_no_data_yet ) 
            throw std::runtime_error("must add observed data before calculating anything\n"); 

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


/**
 * @brief split data thread pool
 * Unlike the above thread pool, this pre-allocates work across the nodes. 
 * "work" is initiated when there is a new object of type dyn_in_t, which is shared across all threads. 
 * Work is performed for every element of the array of static_in_elem_t. The same function 
 * is applied to every pair.
 * @tparam dyn_in_t the type of input that changes every call to work() (e.g. an Eigen::Matrix of a time series observation)
 * @tparam static_in_elem_t (e.g. a particle filter model type)
 * @tparam out_t (e.g. a vector of Eigen::MatrixXd)
 * @tparam num_static_elem the size of the array
 */
template<typename dyn_in_t, typename static_in_elem_t, typename out_t, size_t num_static_elems >
class split_data_thread_pool
{
private: 

    /* type alias for the type of function this thread pool owns */ 
    using comp_func_t    = std::function<out_t(const dyn_in_t&, static_in_elem_t&)>;  
    using agg_func_t     = std::function<out_t(const out_t&, const out_t&)>; 
    using reset_func_t   = std::function<out_t(void)>;
    using final_func_t   = std::function<out_t(const out_t&)>;

    /* mutex */
    mutable std::mutex m_ave_mut;

    /* the number of threads being used in this */
    unsigned m_num_threads;

    /* flag for if pool is operational */
    std::atomic_bool m_done;

    /* flag for if there is a new input */
    std::map<std::thread::id, bool> m_has_new_dyn_input;
    
    /* the accumulated variable (working average) */
    out_t m_working_agg;

    /* promised output of an average */
    std::promise<out_t> m_out;

    /* our compute function */
    comp_func_t m_comp_f;

    /* our aggregating function. First arg is working aggregate, second arg is the value being "added" in*/
    agg_func_t m_agg_f; 

    /* function that resets the average/aggregate quantity*/
    reset_func_t m_reset_f;

    /* a "finalizer" function...usually the identity function, but required for special cases (e.g. if aggregation is summation on integers but you want to average */
    final_func_t m_final_f;

    /* the input common to all o the function, which occasionally gets changed by a single writer  */
    dyn_in_t m_dynamic_input;

    /* the container of a bunch of particle filter objects */
    std::array<static_in_elem_t, num_static_elems>& m_static_input;

    /* the raw threads */
    std::vector<std::thread> m_threads;
    
    /* the RAII thread killer */
    join_threads m_joiner;
   
    /* key: thread id; val: a vector if indexes to iterate over to get work */ 
    std::map<std::thread::id, std::vector<unsigned>> m_work_schedule;


    /**
     * @brief This function runs on all threads, and continuously waits for work to do. When a new input comes, 
     * calculations begin to be performed, and all their outputs are averaged together.
     */ 
    void worker_thread() {

        while(!m_done)
        {

            auto iter = m_has_new_dyn_input.find(std::this_thread::get_id());
            bool is_constructed_and_has_work = (iter != m_has_new_dyn_input.end()) && (iter->second);
            if(is_constructed_and_has_work ){
              
                // **intra-thread**
                // call the work function on each element of the static/stable with the dynamic input shared across all calls
                std::vector<unsigned> work_indexes = m_work_schedule.at(std::this_thread::get_id());
                out_t this_threads_ave = m_reset_f();
                for(const auto& idx : work_indexes){
                    this_threads_ave = m_agg_f(
                            this_threads_ave, 
                            m_comp_f(m_dynamic_input, m_static_input[idx]));
                }

                // **inter-thread** averaging
      	        std::lock_guard<std::mutex> ave_lock{m_ave_mut};
                m_working_agg = m_agg_f(m_working_agg, this_threads_ave);
                m_has_new_dyn_input.at(std::this_thread::get_id()) = false;
                

                // finalize average if you're the last thread
                if( (--m_work_schedule.end())->first  == std::this_thread::get_id() ){
                    m_working_agg = m_final_f(m_working_agg);
                    m_out.set_value(m_working_agg);
                }             

            }else{
                std::this_thread::yield();
            }
        }
    }


public:
  

    /**
     * @brief The ctor spawns the working threads and gets ready to start doing work.
     */  
    split_data_thread_pool(
            std::array<static_in_elem_t, num_static_elems>& static_container,
            comp_func_t comp_f, 
            agg_func_t agg_f, 
            reset_func_t reset_f, 
            final_func_t final_f = [](const out_t& o){ return o;}, 
            bool mt = true) 
        : m_done(false)
        , m_comp_f(comp_f)
        , m_agg_f(agg_f)
        , m_reset_f(reset_f)
        , m_final_f(final_f)
        , m_static_input(static_container)
        , m_joiner(m_threads)
    {  
        unsigned nt = std::thread::hardware_concurrency(); 
        m_num_threads = ((nt > 1) && mt) ? nt : 1;
        if(m_num_threads > 1) m_num_threads -= 1; // one fewer 
        
        try {
            // create threads
            std::thread::id most_recent_id;
            for(unsigned i=0; i< m_num_threads; ++i) {
                m_threads.push_back( std::thread(&split_data_thread_pool::worker_thread, this));
                most_recent_id = m_threads.back().get_id();
                m_work_schedule.insert(std::pair<std::thread::id, std::vector<unsigned> >(most_recent_id, std::vector<unsigned>{}));
                m_has_new_dyn_input.insert(std::pair<std::thread::id, bool>(most_recent_id, false));
            }

            // distribute work as evenly as possibel
            for(size_t i = 0; i < num_static_elems; ++i){
                unsigned thread_idx = i % m_num_threads; 
                std::thread::id thread_id = m_threads[thread_idx].get_id();
                m_work_schedule.at(thread_id).push_back((unsigned)i);
            }

        } catch(...) {
            m_done=true;
            throw;
        }
        
        // initailize working agg
        m_working_agg = m_reset_f();
    }


    /**
     * @brief destructor
     */
    ~split_data_thread_pool() {
        m_done=true;
    }


    /**
     * @brief changes the shared data member, then resets some variables, then starts all the work and returns the "average".
     * @param the new parameter input 
     * @return a floating point average 
     */
    out_t work(dyn_in_t new_input) {

        // set the input for all the threads
        m_dynamic_input = new_input;

        // refresh the aggregate quantity (probably setting it to 0)
        {
        std::unique_lock<std::mutex> ave_lock(m_ave_mut);
        m_working_agg = m_reset_f();
        }

        // start doing work and wait for output to be "returned"
        m_out = std::promise<out_t>();
        for(auto & [key,val] : m_has_new_dyn_input)
            val = true;


        return m_out.get_future().get();
    }
};

#endif // THREAD_POOL_H
