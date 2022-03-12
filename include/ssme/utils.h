#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Dense>
#include <fstream>
#include <array>
#include <iostream> // std::cerr
#include <fstream>  // std::ofstream
#include <string>   // string
#include <vector>   // vector


/**
 * @namesapce utils
 */
namespace utils{


    /**
     * @brief reads in data in a csv file with no header and separated by commas.
     * @tparam nc the number of columns.
     * @param file_loc the string filepath of the file.
     * @return a std::vector of your data. Each elemenet is a row in Eigen::Vector form.
     */
    template<size_t nc, typename float_t>
    std::vector<Eigen::Matrix<float_t,nc,1> > read_data(const std::string& file_loc)
    {
        // returning this. gotta build it up
        std::vector<Eigen::Matrix<float_t,nc,1> > data;
        
        // start reading
        std::string line;
        std::ifstream ifs;
        ifs.open(file_loc);
        std::string one_number;    
        if(!ifs.is_open()){     // check if we can open inFile
            std::cerr << "utils::read_data() failed to read data from: " << file_loc << "\n";
        }
        
        // didn't fail...keep going
        while ( std::getline(ifs, line) ){     // get a whole row as a string
        
            std::vector<float_t> data_row;
            try{
                // get a single element on a row
                std::istringstream buff(line);
                
                // make one number between commas
                while(std::getline(buff, one_number, ',')){ 
                    data_row.push_back(std::stod(one_number));
                }
                
            } catch (const std::invalid_argument& ia){
                std::cerr << "Invalid Argument: " << ia.what() << "\n";
                continue;
            }   
            
            // now append this Vec to your collection
            Eigen::Map<Eigen::Matrix<float_t,nc,1> > drw(&data_row[0], nc);
            data.push_back(drw);
        }
        
        return data;
    } 
    
    
    /**
 * @class csv_param_sampler
 * @author taylor
 * @file utils.h
 * @brief Class that samples parameters from a csv file. File must not have header, and be comma separated.
 * @tparam dimparam the dimension of a parameter vector (also the number of columns in the csv file)
 * @tparam float_t the type of floating point
 */
template<size_t dimparam, typename float_t>
class csv_param_sampler
{
public:

    /** "param sized vector" type alias for linear algebra stuff **/
    using psv         = Eigen::Matrix<float_t, dimparam,1>;

    
    /**
     * @brief Default constructor is deleted because you need a file location. 
     */
    csv_param_sampler() = delete; 

    /**
     * @brief ctor 1
     * @param the name of the csv of parameter samples (must be headerless and comma separated)
     */
    explicit csv_param_sampler(const std::string &param_csv_filename);


    /**
     * @brief constructor with seed
     * @param the name of the csv of parameter samples (must be headerless and comma separated)

     */
    csv_param_sampler(const std::string &param_csv_filename, unsigned long seed);
    
    
    /**
     * @brief draw random (re-)sample parameter vector
     */
    psv samp();


private:

    std::mt19937 m_gen; 
    std::uniform_int_distribution<int> m_idx_sampler;
    std::vector<psv> m_param_samps;
};


template<size_t dimparam, typename float_t>
csv_param_sampler<dimparam,float_t>::csv_param_sampler(const std::string &param_csv_filename)
    : m_gen{static_cast<std::uint32_t>(
                    std::chrono::high_resolution_clock::now().time_since_epoch().count() )}
{
    m_param_samps = read_data<dimparam,float_t>(param_csv_filename);
    m_idx_sampler = std::uniform_int_distribution<int>(0, m_param_samps.size() - 1);
}


template<size_t dimparam, typename float_t>
csv_param_sampler<dimparam,float_t>::csv_param_sampler(const std::string &param_csv_filename, unsigned long seed)
    : m_gen{static_cast<std::uint32_t>(seed)}
{
    m_param_samps = read_data<dimparam,float_t>(param_csv_filename);
    m_idx_sampler = std::uniform_int_distribution<int>(0, m_param_samps.size() - 1);
}


template<size_t dimparam, typename float_t>
auto csv_param_sampler<dimparam,float_t>::samp() -> psv
{
    return m_param_samps[m_idx_sampler(m_gen)];
}

} // namespace utils


#endif // UTILS_H
