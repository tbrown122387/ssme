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
#include <iostream>


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
        for(auto & m_thread : m_threads) {
            if(m_thread.joinable())
                m_thread.join();
        }
    }
};


/**
 * @brief Here we have many concurrent parameter readers, but only a single parameter writer.
 * This thread pool owns one function that returns one (random) floating point number. The pool sits ready to perform calculations
 * on any new parameter value. Once a new parameter value is received, this pool calls its function a fixed number of times, 
 * and all of the function output is averaged in a thread-safe way. Actually, these function evals are expected to be in the log
 * space, and the log-mean-exp is calculated using the log-sum-exp trick. For our particular applications, this function will also depend
 * on observed data that doesn't change once the thread pool has been initialized. Finally, note that this function is assumed not to 
 * allocate memory on the heap. 
 * @tparam dyn_data_t dynamic data type. The type of input that gets changed repeatedly.
 * @tparam static_data_t the type of input that only gets set once
 * @tparam func_output_t type of function output
 * @tparam debug do you want debug printing messages?
 */
template<typename dyn_data_t, typename static_data_t, typename func_output_t, bool debug = false>
class thread_pool
{
public:  

    /* type alias for the type of function this thread pool owns */ 
    using F = std::function<func_output_t(dyn_data_t, static_data_t)>; 

private:
    static_assert( std::is_floating_point<func_output_t>::value,  
                   "function output type must be floating point");


    /* shared lock for reading in new parameter values */  
    mutable std::shared_mutex m_input_mut; 

    /* locks the average and count of comps */
    mutable std::shared_mutex m_output_mut;

    /* number of threads in pool */
    unsigned m_num_threads;
 
    /* the number of calls that have been completed so far. less than full count signals to continue working */
    std::atomic_uint m_work_start_count;

    /* flag to signal that finalization is in order */
    std::atomic_uint m_work_done_count;

    /* flag for if pool is operational */
    std::atomic_bool m_done;

    /* the unchanging number of function calls you want for each fresh new input */
    const unsigned m_total_calcs;
   
    /* store all the calculations in a vector */ 
    std::vector<func_output_t> m_all_calcs;

    /* promised output of an average */
    std::promise<func_output_t> m_out;

    /* our function that gets called every time */
    F m_f; 

    /* input 1 of 2: dynamic parameter vector that is used as an input to the function, which occasionally gets changed by a single writer  */
    dyn_data_t m_param;

    /* input 2 of 2: unchanging observed data that is used as an input to the function */ 
    static_data_t m_observed_data; 
 
    /* whether the observed "static" data has been added to the thread pool yet */ 
    bool m_no_data_yet; 
   
    /* the raw threads */
    std::vector<std::thread> m_threads;
    
    /* the RAII thread killer (defined above) */
    join_threads m_joiner;

public:

    thread_pool() = delete;  

    /**
     * @brief The ctor spawns the working threads and gets ready to start doing work.
     * @param f the function that gets called a bunch of times.
     * @param num_comps the number of times to call f with each new parameter input.
     * @param mt do you want multiple threads?
     * @param num_threads the number of threads you want or 0 which decides what's available for you
     */  
    thread_pool(F f, unsigned num_comps, bool mt = true, unsigned num_threads = 0)
        : m_num_threads(num_threads)
        , m_work_start_count(num_comps+1) 
        , m_work_done_count(0)
        , m_done(false)
        , m_total_calcs(num_comps) 
        , m_f(f)
        , m_no_data_yet(true) 
        , m_joiner(m_threads)

    {

        // assign work load and start threads, getting them ready to work
        if ( m_num_threads == 0 ) {
            if( mt ) {
                unsigned nt  = std::thread::hardware_concurrency();
                m_num_threads = (nt > 1) ? nt : 1;
            }else {
                m_num_threads = 1;
            }
        }

        // check for disagreement
        if ((m_num_threads > 1) && (!mt)) {
            throw std::invalid_argument("can't request single threaded and multiple threads at the same time!");
        }else if( mt && (m_num_threads == 1)) {
            throw std::runtime_error("requested multiple threads but only one is available");
        }

        if constexpr(debug)
            std::cout << "launching " << m_num_threads << " threads\n";

        // launch all threads
        try {
            for(unsigned i=0; i < m_num_threads; ++i) {
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
        if( !m_no_data_yet )
            throw std::runtime_error("you already called add_observed_data once before!");
        std::unique_lock<std::shared_mutex> input_lock(m_input_mut);
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

        // reset output sum 
        {
            std::unique_lock<std::shared_mutex> output_lock(m_output_mut);
            if constexpr(debug)
                std::cout << "cleraing output\n";
            m_out = std::promise<func_output_t>();
            m_all_calcs.clear();
        }

        // set new work and signal threads to begin
        {
            std::unique_lock<std::shared_mutex> param_lk(m_input_mut);
            if constexpr(debug)
                std::cout << "resetting work input now \n";
            m_param = new_param;
        }
        m_work_done_count = 0;
        m_work_start_count = 0; 

        // wait for work to come back and then return it
        return m_out.get_future().get();
    }

private:


    /**
     * @brief This function runs on all threads, and continuously waits for work to do. When a new parameter comes,
     * calculations begin to be performed, and all their outputs are averaged together. When work is finished, threads idle
     */
    void worker_thread() {

        if constexpr(debug) {
            std::unique_lock<std::shared_mutex> param_lock(m_input_mut);
            std::cout << "DEBUG: starting worker_thread()\n";
        }

        while (!m_done) {

            
            // get number of times work has been done
            unsigned calc_count = m_work_start_count++;
                
            // either finalize the average, perform difficult work, or spin
            if (calc_count < m_total_calcs) {
    
                // call the expensive function that returns an approximate log-likelihood
                func_output_t val;
                {
                    std::shared_lock<std::shared_mutex> param_lock(m_input_mut);
                    val = m_f(m_param, m_observed_data);
                }

                // store all calcs
                {
                    std::unique_lock<std::shared_mutex> out_lock{m_output_mut};
                    if constexpr(debug)
                        std::cout << "writing out a " << val <<  "in space " << calc_count << "\n";
                    m_all_calcs.push_back(val);
                }
                m_work_done_count++;

            } else {
               
                // test and increment so it only does thi sonce
                unsigned expected = m_total_calcs; 
                bool need_finalization = m_work_done_count.compare_exchange_strong(expected,0);
                if(need_finalization){
                
                    func_output_t m = *max_element(m_all_calcs.begin(), m_all_calcs.end());
                    func_output_t sum_exp = 0;
                    for(unsigned i = 0; i < m_total_calcs; ++i){
                        sum_exp += std::exp(m_all_calcs[i] - m);
                    }
                    func_output_t final_val = m + std::log(sum_exp) - std::log(m_total_calcs);
    
                    if constexpr(debug)
                        std::cout << "finalizing " << final_val << "\n";
                    m_out.set_value(final_val);
    
                }else{
                    std::this_thread::yield();
                }
            }
        }
    }
};


/**
 * @brief split data thread pool
 * Unlike the above thread pool, this pre-allocates work across the nodes. 
 * "work" is initiated when there is a new object of type dyn_in_t, which is shared across all threads. 
 * Work is performed for every element of the array of static_in_elem_t. The same function 
 * is applied to every pair. Keep in mind that this function should not allocate memory on the heap!
 * @tparam dyn_in_t the type of input that changes every call to work() (e.g. an Eigen::Matrix of a time series observation)
 * @tparam static_in_elem_t (e.g. a particle filter model type)
 * @tparam out_t (e.g. a vector of Eigen::MatrixXd)
 * @tparam num_static_elem the size of the array
 * @tparam debug true if you want debug messages printed
 */
template<typename dyn_in_t, typename static_in_elem_t, typename out_t, size_t num_static_elems, bool debug = false>
class split_data_thread_pool
{

private: 

    /* type aliases for the type of functions this thread pool owns */
    using comp_func_t      = std::function<out_t(const dyn_in_t&, static_in_elem_t&)>;
    using inter_agg_func_t = std::function<out_t(const out_t&, const out_t&, unsigned num_threads, unsigned num_terms_in_thread)>; 
    using intra_agg_func_t = std::function<out_t(const out_t&, const out_t&, unsigned num_terms_in_thread)>; 
    using reset_func_t     = std::function<out_t(void)>;
    using final_func_t     = std::function<out_t(const out_t&)>;

    /* mutex to write out to inter-thread average and average counter */
    mutable std::mutex m_output_mut;

    /* mutex that to read/write dynamic input */
    mutable std::shared_mutex m_input_mut;

    /* the number of threads being used in this */
    unsigned m_num_threads;

    /* flag for if pool is operational */
    std::atomic_bool m_done;

    /* flag for if there is a new input...signals a thread to start working on new input*/
    std::map<std::thread::id, std::atomic_bool> m_has_new_dyn_input;
    
    /* the accumulated variable (working average) */
    out_t m_working_agg;

    /* promised output of an average */
    std::promise<out_t> m_out;

    /* our compute function */
    comp_func_t m_comp_f;

    /* our aggregating function. First arg is working aggregate, second arg is the value being "added" in*/
    inter_agg_func_t m_inter_agg_f; 

    /* our aggregating function. First arg is working aggregate, second arg is the value being "added" in*/
    intra_agg_func_t m_intra_agg_f; 

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

    /* atomic counter for how many pieces of work have been done...full count signals finalization is ready */
    std::atomic_uint m_num_thread_aves_done;

public:
  

    /**
     * @brief The ctor spawns the working threads and gets ready to start doing work.
     * @param static_container unchanging data used to perform calculations
     * @param comp_f performs expensive calculation
     * @param inter_agg_f adds thread's work to overall average
     * @param intra_agg_f adds to intra-thread average
     * @param reset_f reset function (usually resets average to 0)
     * @param final_f finalizer function (default is identity)
     * @param mt true if you want multithreading
     * @param num_threads
     */  
    split_data_thread_pool(
            std::array<static_in_elem_t, num_static_elems>& static_container,
            comp_func_t comp_f, 
            inter_agg_func_t inter_agg_f,
            intra_agg_func_t intra_agg_f,
            reset_func_t reset_f, 
            final_func_t final_f = [](const out_t& o){ return o;}, 
            bool mt = true,
            unsigned num_threads = 0)
        : m_num_threads(num_threads)
        , m_done(false)
        , m_comp_f(comp_f)
        , m_inter_agg_f(inter_agg_f)
        , m_intra_agg_f(intra_agg_f)
        , m_reset_f(reset_f)
        , m_final_f(final_f)
        , m_static_input(static_container)
        , m_joiner(m_threads)
        , m_num_thread_aves_done(0)
    {


        // assign work load and start threads, getting them ready to work
        if ( m_num_threads == 0 ) {
            if( mt ) {
                unsigned nt  = std::thread::hardware_concurrency();
                m_num_threads = (nt > 0) ? nt : 1;
            }else {
                m_num_threads = 1;
            }
        }

        // check for disagreement
        if ((m_num_threads > 1) && (!mt)) {
            throw std::invalid_argument("can't request single threaded and multiple threads at the same time!");
        }else if( mt && (m_num_threads == 1)) {
            throw std::runtime_error("requested multiple threads but only one is available");
        }

        if constexpr(debug)
            std::cout << "launching " << m_num_threads << " threads\n";


        // launch threads
        try {
            for(unsigned i=0; i< m_num_threads; ++i) 
                m_threads.push_back( std::thread(&split_data_thread_pool::worker_thread, this));
        } catch(...) {
            m_done=true;
            throw;
        }

        // for all threads, set work schedule, and work signaler
        {
            std::unique_lock<std::shared_mutex> input_lock(m_input_mut);
            std::thread::id thread_id;
            for(unsigned i = 0; i < m_num_threads; ++i){

                thread_id = m_threads[i].get_id();
                m_work_schedule.insert(
                        std::pair<std::thread::id, std::vector<unsigned> >(
                            thread_id, std::vector<unsigned>{}));
                m_has_new_dyn_input.insert(
                        std::pair<std::thread::id, bool>(
                            thread_id, false));
            }

            for(unsigned i = 0; i < num_static_elems; ++i){
                unsigned thread_idx = i % m_num_threads;
                thread_id = m_threads[thread_idx].get_id();
                m_work_schedule.at(thread_id).push_back((unsigned)i);
            }
        }

        if constexpr(debug){
            std::unique_lock<std::mutex> param_lock(m_output_mut);
            std::cout << "DEBUG: split_data_thread_pool() initiate threads, assign work sched\n";
        }

    }


    /**
     * @brief destructor
     */
    ~split_data_thread_pool() {
        m_done=true;
    }


    /**
     * @brief changes the shared data member, then resets some variables, then starts all the work and returns the "average".
     * @param new_input new parameter input
     * @return a floating point average 
     */
    out_t work(dyn_in_t new_input) {
       
        // refresh the output quantity (probably setting it to 0)
        {
            std::unique_lock<std::mutex> output_lock(m_output_mut);
            m_out = std::promise<out_t>();
            m_working_agg = m_reset_f();
        }
        
        // set the input for all the threads
        m_num_thread_aves_done = 0;
        {
            std::unique_lock<std::shared_mutex> input_lock(m_input_mut);
            m_dynamic_input = new_input;
            for(auto & [key,val] : m_has_new_dyn_input)
                val = true;
        }

        if constexpr(debug){
            std::unique_lock<std::mutex> param_lock(m_output_mut);
            std::cout << "DEBUG: work() sets new dynamic input, resets output, and signals to all threads there is new work.\n";
        }

        // wait for work to finish and then return result
        return m_out.get_future().get();
    }


private:

    /**
     * @brief This function runs on all threads, and continuously waits for work to do. When a new input comes, 
     * calculations begin to be performed, and all their outputs are averaged together. When all the threads have
     * finished their work, the final thread performs finalization.
     */ 
    void worker_thread() {

        if constexpr(debug) {
            std::unique_lock<std::mutex> param_lock(m_output_mut);
            std::cout << "DEBUG: starting worker_thread()\n";
        }

        while (!m_done) {



            // create some variables used to decide what to do
            bool this_thread_has_work = false; 
            {
                std::shared_lock<std::shared_mutex> read_input_lock(m_input_mut);
                auto iter = m_has_new_dyn_input.find(std::this_thread::get_id());
                this_thread_has_work = (iter != m_has_new_dyn_input.end()) && (iter->second);
            }
 
            // make it impossible for this same thread to redo work
            // thread safe because each map element corresponds with unique thread
            if(this_thread_has_work){
                m_has_new_dyn_input.at(std::this_thread::get_id()) = false;
            }

            if constexpr(debug) {
                std::unique_lock<std::mutex> param_lock(m_output_mut);
                std::cout << "DEBUG:  this_thread_has_work: " << this_thread_has_work << "\n";
            }


            // either do computations, finalize intra-thread average, or spin
            if ( this_thread_has_work ) { 

                // **intra-thread**
                // call the work function on each element of the static/stable with the dynamic input shared across all calls
                std::vector<unsigned> work_indexes = m_work_schedule.at(std::this_thread::get_id());
                out_t this_threads_ave = m_reset_f();
                for (const auto &idx: work_indexes) {
                    this_threads_ave = m_intra_agg_f(
                            this_threads_ave,
                            m_comp_f(m_dynamic_input, m_static_input[idx]),
                            work_indexes.size());
                    if constexpr(debug) {
                        std::unique_lock<std::mutex> param_lock(m_output_mut);
                        std::cout << "DEBUG: thread " << std::this_thread::get_id() << "'s average is updated to: "
                            << this_threads_ave.second << "\n";
                    }
                }
                
                {
                    std::lock_guard<std::mutex> output_lock{m_output_mut};
                    m_working_agg = m_inter_agg_f(m_working_agg,
                            this_threads_ave,
                            m_num_threads,
                            work_indexes.size());
                }

                // help signal that finalization is ready
                m_num_thread_aves_done++;

            } else { // no work so we check if we can finalize, and if not, we idle

                // test if we can finalize
                // if it does, need to increment num_thread_aves_done immediately so other threads don't try to do the same!
                unsigned int num_threads = m_num_threads; // need a copy because it'll get modified
                bool time_to_finalize_output = m_num_thread_aves_done.compare_exchange_strong(num_threads, num_threads+1);

                if constexpr(debug) {
                    std::unique_lock<std::mutex> param_lock(m_output_mut);
                    std::cout << "DEBUG: thread " << std::this_thread::get_id() << " is attempting to set the final value to the future: " << time_to_finalize_output << "\n";
                }

                if( time_to_finalize_output){                
                    //std::lock_guard<std::mutex> output_lock{m_output_mut};
                    //don't need lock becuase threads have to be done with m_working_agg in order to increment count
                    m_working_agg = m_final_f(m_working_agg);
                    m_out.set_value(m_working_agg);
                }else{
                    std::this_thread::yield();
                }
            }
        }
    }
};

#endif // THREAD_POOL_H
