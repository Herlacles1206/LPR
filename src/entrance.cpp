#include "plate_locate.h"
#include "chars_segment.h"
#include "chars_identify.h"
#include "chars_recognise.h"
#include "config.h"
#include "core_func.h"
#include "plate_recognize.h" 
#include "plate.hpp"
#include "entrance.h"

using namespace std;
using namespace cv;
using namespace easypr;



void Entrance::print_easypr(){
	cout<<"wellcom to easypr!"<<endl;
	return;
}
//车牌字符分割
int Entrance::test_chars_segment( const char *img_path  ) {
  std::cout << "test_chars_segment" << std::endl;

 // cv::Mat src = cv::imread("image/chars_segment.jpg");
	cout<<string(img_path)<<endl;
  cv::Mat src = cv::imread(string(img_path));  
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

int Entrance::test_chars_identify() {
  std::cout << "test_chars_identify" << std::endl;

  cv::Mat plate = cv::imread("resources/image/chars_identify.jpg");

  std::vector<Mat> matChars;
  std::string license;

  easypr::CCharsSegment cs;

  int result = cs.charsSegment(plate, matChars);
  cout<<"test:::"<<endl;
  if (result == 0) {
    for (auto block : matChars) {
      auto character = easypr::CharsIdentify::instance()->identify(block);
      license.append(character.second);
    }
  }

  std::string plateLicense = "苏E771H6";
#ifdef OS_WINDOWS
  plateLicense = utils::utf8_to_gbk(plateLicense.c_str());
#endif
  std::cout << "plateLicense: " << plateLicense << std::endl;
  std::cout << "plateIdentify: " << license << std::endl;

  if (plateLicense != license) {
    std::cout << "Identify Not Correct!" << std::endl;
    return -1;
  }
  std::cout << "Identify Correct!" << std::endl;

  return result;
}

int Entrance::test_chars_recognise() {
  std::cout << "test_chars_recognise" << std::endl;

  cv::Mat src = cv::imread("resources/image/chars_recognise.jpg");
  easypr::CCharsRecognise cr;

  std::string plateLicense = "";
  int result = cr.charsRecognise(src, plateLicense);
  if (result == 0)
    std::cout << "charsRecognise: " << plateLicense << std::endl;
  return 0;
}

//车牌定位
int Entrance::test_plate_locate(const char *type, const char *img_path) {
	
  cout << "test_plate_locate" << endl;
  const string file = String(img_path);
//cout<<"type:"<<type[0]<<endl;


  cv::Mat src = imread(file);

  vector<cv::Mat> resultVec;
  vector<easypr::CPlate> candPlates;
  easypr::CPlateLocate plate;
	int result = -1;
	//选择不同locate方法
	if(type[0]=='c') result =plate.plateColorLocate(src, candPlates);
	else if(type[0]=='s') result =plate.plateSobelLocate(src, candPlates);
	else if(type[0]=='m') result =plate.plateMserLocate(src, candPlates);
	else if(type[0]=='l') result =plate.plateLocate(src, candPlates);
	else {
		int result = -1;
		cout<<"输入有错！"<<endl;
	}
//cout<<"result:"<<result<<endl;
  if (result == 0) {
	size_t num = candPlates.size();
	cout<<"num"<<num<<endl;
	
    for (size_t j = 0; j < num; j++) {
      cv::Mat resultMat = candPlates[j].getPlateMat();
      imshow("m_plate_locate", resultMat);
      waitKey(0);
    }
    destroyWindow("plate_locate");
  }

  return result;
}

void Entrance::run(const char *img_path){
	
	const string file = String(img_path);
	easypr::CPlateRecognize pr;

	//设置pr属性
	pr.setResultShow(true);
	pr.setDetectShow(true);
	//pr.setDetectType(PR_DETECT_CMSER);//设置EasyPR采用的车牌定位算法
	pr.setDetectType(PR_DETECT_COLOR | PR_DETECT_SOBEL);//设置EasyPR采用的车牌定位算法
	//pr.setDetectType(PR_DETECT_COLOR);
	pr.setLifemode(true);//这句话设置开启生活模式，这个属性在定位方法为SOBEL时可以发挥作用，能增大搜索范围，提高鲁棒性。
	pr.setMaxPlates(4);
	
	vector<std::string> plateVec;
	std::string plateLicense = "";
	Mat src = imread(file);
	int result = pr.plateRecognize(src, plateVec);
	if(result == 0){
		//cout<<"车牌："<<plateLicense<<endl;
		
		size_t num = plateVec.size();
		cout<<"num"<<num<<endl;
		for(size_t i = 0; i < num; i++){
			cout<<"车牌："<<plateVec.at(i)<<endl;
		}

	}
	

}

void Entrance::plate(const char *img_path){
	const string file = String(img_path);
	easypr::CPlateRecognize pr;	
	easypr::CCharsRecognise cr;
	std::string plateLicense = "";
	cv::Mat src = imread(file);
	vector<easypr::CPlate> candPlates;
 	easypr::CPlateLocate plate;
//首选color，如果不行再用别的
	int result = plate.plateColorLocate(src, candPlates);
	
	if(candPlates.size()==0) result = result =plate.plateLocate(src, candPlates);
	//int result = plate.plateColorLocate(src, candPlates);
	//int result =plate.plateMserLocate(src, candPlates);
	//int result =plate.plateMserLocate(src, candPlates);
	//int result =plate.plateLocate(src, candPlates);	
	if (result == 0) {
		size_t num = candPlates.size();	
		for (size_t j = 0; j < num; j++) {
		cv::Mat resultMat = candPlates[j].getPlateMat();
		int result = cr.charsRecognise(resultMat, plateLicense);
		if (result == 0)
	 		std::cout << "charsRecognise: " << plateLicense << std::endl;

		imshow("plate_locate", resultMat);
		waitKey(0);
		}
	destroyWindow("plate_locate");
  	}
	return;

}

/*
int main(int argc, char **argv)
{

	clock_t start,ends;
	 start=clock();
	if(argc < 3){
		fprintf(stderr, "usage:%s %s %s   [./myeasypr] [run/seg/locate(color_locate,sobel_locate,mser_locate)/identify/recognise] [imgage_path]\n", argv[0], argv[1],argv[2]);
		return 0;
	}
	
	if(strcmp(argv[1],"seg")==0 ) test_chars_segment(argv[2]);

	else if(strcmp(argv[1],"run")==0) run(argv[2]);		
		
	else if( strcmp(argv[1],"color_locate")==0) test_plate_locate(argv[1],argv[2]);
	else if( strcmp(argv[1],"sobel_locate")==0) test_plate_locate(argv[1],argv[2]);
	else if( strcmp(argv[1],"mser_locate")==0) test_plate_locate(argv[1],argv[2]);
	else if( strcmp(argv[1],"locate")==0) test_plate_locate(argv[1],argv[2]);

	else if( strcmp(argv[1],"plate")==0) plate(argv[2]);
 		
	else if( strcmp(argv[1],"identify")==0) test_chars_identify();
 		
	else if( strcmp(argv[1],"recognise")==0) test_chars_recognise();
 	
	else{
		fprintf(stderr, "usage:%s %s %s   [./myeasypr] [seg/locate/identify/recognise] [imgage_path]\n", argv[0], argv[1],argv[2]);
		return 0;
		
	}
	ends=clock();
 	cout<<"耗时："<<((float)(ends-start))/CLOCKS_PER_SEC<<"秒"<<endl; 
	return 0;
}
*/

  
