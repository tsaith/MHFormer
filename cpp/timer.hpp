#include "opencv2/opencv.hpp"
#include <ctime>

using namespace cv;


clock_t getClock() {

    clock_t now = clock();

#ifdef __linux__
    now = now / 1000;
#endif

    return now;

}


class Timer 
{
public:
    void tic()
    {
        count_begin = getTickCount();
    };
    void toc()
    {
        count_end = getTickCount();
        count_delta = count_end - count_begin;
        fps = getTickFrequency()/count_delta;
        dt = 1.0/fps;
    };

    void delay(int dt) // dt is in milliseconds
    { 
        clock_t now = getClock(); 

        while(getClock() - now < dt); 
    } 

    double get_fps() { return fps;};
    double get_dt() { return dt; };

private:
    int64 count_begin = 0;
    int64 count_end = 0;
    int64 count_delta = 0;

    float fps = 0.0;
    float dt = 0.0;

};
