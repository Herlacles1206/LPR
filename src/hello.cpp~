#include "hello.h"
#include "plate_segment.h"

using namespace std;
using namespace cv;
using namespace easypr;
int show_segment_result() {
	std::cout << "test_chars_segment" << std::endl; 

	cv::Mat src = cv::imread("/home/himon/c-Projects/plate-seg/chars_recognise.jpg");

	std::vector<cv::Mat> resultVec;
	easypr::CCharsSegment plate;

	int result = plate.charsSegment(src, resultVec);
	if (result == 0) {
		size_t num = resultVec.size();
		for (size_t j = 0; j < num; j++) {
			cv::Mat resultMat = resultVec[j];
			cv::imshow("chars_segment", resultMat);
			cv::waitKey(0);
		}
		cv::destroyWindow("chars_segment");
	}

	return result;
	
}
void show_img(){

    Mat img = imread("/home/himon/code/yolo/car.jpg", CV_LOAD_IMAGE_COLOR);
    namedWindow( "lena", CV_WINDOW_AUTOSIZE );
    imshow("lena", img);
    waitKey(0);
}
void print(int i)
{
    cout<<"hello c link c++ "<<i<<endl;
}

