#ifndef CONFIG_H
#define CONFIG_H

#include "myslam/common_include.h" 

namespace myslam
{
    class Config
    {
    private:
        static std::shared_ptr<Config> config_;
        //cv::FileStorage file_;
        YAML::Node yamlConfig_;

        Config() {} // private constructor makes a singleton
    public:
        ~Config();  // close the file when deconstructing 

        // set a new config file 
        static void setParameterFile(const std::string& filename);

        // access the parameter values
        template< typename T >
        static T get(const std::string& key)
        {
            /*T res = (T)(config_->file_[key]);
            return res;*/
            return Config::config_->yamlConfig_[key].as<T>();
        }
    };
}

#endif // CONFIG_H