#pragma once

#include <opencv2/opencv.hpp>

//SGBM变量初始化
static cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create();

//图像读取
int imgRead(std::string path_left, std::string path_right);

//极线校正参数初始化
void Polar_Init();

//极线校正计算
void Polar_Compute();

//SGBM视差立体匹配
void SGBM_Match(int pos, void* userdata);

//SGBM参数调整框
void SGBM_Window();

//图像拼接
void Img_Merge(int gap);
    
//极线矫正参数
struct Polar_Option
{
    cv::Mat cameraMatrixL;  //左侧相机内参矩阵
    cv::Mat cameraMatrixR;  //右侧相机内参矩阵

    cv::Mat distCoeffL;     //左侧畸变参数
    cv::Mat distCoeffR;     //右侧畸变参数

    cv::Mat R2L_R;          //右侧相机到左侧相机的旋转矩阵
    cv::Mat R2L_T;          //右侧相机到左侧相机的平移矩阵

    cv::Mat RL;             //左相机的矫正变换矩阵
    cv::Mat RR;             //右相机的矫正变换矩阵

    cv::Mat PL;             //左相机的新坐标系下的投影矩阵
    cv::Mat PR;             //右相机的新坐标系下的投影矩阵

    cv::Mat Q;              //深度差异映射矩阵

    cv::Mat rmap[2][2];      //rmap函数用到的参数
    cv::Size image_size;
};

//极线矫正参数
static Polar_Option Option1;

//左侧图像输入
static cv::Mat img_left;

//右侧图像输入
static cv::Mat img_right;

//极线矫正后左图像输出
static cv::Mat left_out;

//极线矫正后右图像输出
static cv::Mat right_out;

//校正后图像拼接对比
static cv::Mat merge;

//视差图
static cv::Mat disp;
