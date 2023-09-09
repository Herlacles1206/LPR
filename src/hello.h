#ifndef CPP_HEADER
#define CPP_HEADER

#include <iostream>
#ifdef OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif

extern "C" {
	void show_img();
	void print(int i);
	int show_segment_result();
} 
#endif CPP_HEADER
