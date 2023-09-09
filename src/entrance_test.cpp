#include "plate_locate.h"
#include "chars_segment.h"
#include "chars_identify.h"
#include "chars_recognise.h"
#include "config.h"
#include "core_func.h"
#include "plate_recognize.h" 
#include "plate.hpp"
#include "entrance_test.h"


#ifdef __cplusplus  
extern "C"{  
#endif  

using namespace std;
using namespace cv;
using namespace easypr;

float my_get_pixel(float *data,int im_h,int im_w,int im_c, int x, int y, int c)
{
    assert(x < im_w && y < im_h && c < im_c);//C语言中的断言
    return data[c*im_h*im_w + y*im_w + x];
	
}

char* plate_recognise_mat(int h,int w,int c,float *data){

 //讲传过来的图像data转为cv::Mat类型	
	for(int i = 0; i < w*h; ++i){
		float swap = data[i];
		data[i] = data[i+w*h*2];
		data[i+w*h*2] = swap;
	}
	Mat src(h,w,CV_8UC3);
	int s0 = src.step[0];
	int s1 = src.step[1];
	//cout<<"step[0]:"<<s0<<"  step[1]:"<<s1<<endl;
	unsigned char *mdata= src.data;
	int x,y,k;
	for(y = 0; y < h; ++y){ //行,h,rows
		for(x = 0; x < w; ++x){//列,w,cols
			for(k= 0; k < c; ++k){//每个像素
				mdata[y*s0 + x*s1 + k] = (unsigned char)(my_get_pixel(data,h,w,c,x,y,k)*255);
			}
		}
	}

	cv::imshow("src", src);
	//string result = plate_mat(src);
	string result = run_mat(src);
	const char *tmp = result.c_str();
	char *p = const_cast<char*>(tmp);
	//cout<<"result:"<<result<<endl;
	
	return p;
}

void print_easypr(){
	cout<<"entrance_test,,wellcom to easypr!"<<endl;
	return;
}
//车牌字符分割
int test_chars_segment( const char *img_path  ) {
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

int test_chars_identify() {
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

int test_chars_recognise() {
  std::cout << "test_chars_recognise" << std::endl;

  cv::Mat src = cv::imread("/home/himon/code/yolo/darknet-master/resources/image/chars_recognise.jpg");

  easypr::CCharsRecognise cr;
  std::string plateLicense = "";
  int result = cr.charsRecognise(src, plateLicense);

  if (result == 0)
    std::cout << "charsRecognise: " << plateLicense << std::endl;
  return 0;
}

//车牌定位
int test_plate_locate(const char *type, const char *img_path) {
	
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

void run(const char *img_path){
	
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


string run_mat(const Mat &src){
	
	easypr::CPlateRecognize pr;

	//设置pr属性
	pr.setResultShow(true);
	pr.setDetectShow(true);
	//pr.setDetectType(PR_DETECT_CMSER);//设置EasyPR采用的车牌定位算法
	pr.setDetectType(PR_DETECT_COLOR | PR_DETECT_SOBEL|PR_DETECT_CMSER);//设置EasyPR采用的车牌定位算法
	//pr.setDetectType(PR_DETECT_COLOR);
	pr.setLifemode(true);//这句话设置开启生活模式，这个属性在定位方法为SOBEL时可以发挥作用，能增大搜索范围，提高鲁棒性。
	pr.setMaxPlates(4);
	
	vector<std::string> plateVec;
	std::string plateLicense = "";
	int result = pr.plateRecognize(src, plateVec);
	if(result == 0){
		cout<<"车牌："<<plateLicense<<endl;
		
		size_t num = plateVec.size();
		//cout<<"num"<<num<<endl;
		for(size_t i = 0; i < num; i++){
			cout<<"车牌："<<plateVec.at(i)<<endl;
		}

	}
	
	return plateLicense;
}


string plate_mat(const Mat &src){	
	easypr::CPlateRecognize pr;	
	easypr::CCharsRecognise cr;
	std::string plateLicense = "";
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
	return plateLicense;

}

void plate(const char *img_path){
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
	return ;

}

#ifdef __cplusplus  
}  
#endif  

