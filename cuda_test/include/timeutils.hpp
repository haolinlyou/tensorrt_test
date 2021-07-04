/**
**********************************************************************************************************************************************************************************************************************************
* @file:	timeutils.hpp
* @author:	lk
* @email:	lk123400@163.com
* @date:	2021-06-23 19:08:29 Wednesday
* @brief:	
**********************************************************************************************************************************************************************************************************************************
**/

#ifndef __TIMEUTILS__H__
#define __TIMEUTILS__H__

#include <iostream>
#include <chrono>



//时间单位
namespace TimeUnit
{
    typedef std::chrono::hours HOURS;
    typedef std::chrono::minutes MINUTES;
    typedef std::chrono::seconds SEC;
    typedef std::chrono::milliseconds MILLISEC;
    typedef std::chrono::milliseconds MICROSEC;
    typedef std::chrono::nanoseconds NANOSEC;
} // namespace TimeUnit




//时间util，和时间相关的定义类都在此处
class TimeUtil
{
private:
    /* data */
    std::chrono::high_resolution_clock::time_point startTime;
public:
    TimeUtil(/* args */);
    ~TimeUtil();
    std::chrono::high_resolution_clock::time_point getCurrentTime() 
    {
        return std::chrono::high_resolution_clock::now();
    }

    //开启定时器，获取定时器开启的起始时间
    void startTimer()
    {
        startTime = getCurrentTime();
    }

    //获取时间段的模板函数，主要用来记录耗时等，模板主要是区别时间单位，TimeUnit类型
    template <typename T>
    uint64_t getDuration()
    {
        T timeInterval = std::chrono::duration_cast<T>(getCurrentTime() - startTime);
        return timeInterval.count();
    }
};

TimeUtil::TimeUtil(/* args */)
{
}

TimeUtil::~TimeUtil()
{
}


#endif  //!__TIMEUTILS__H__
