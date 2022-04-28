#pragma once

#include <opencv2/opencv.hpp>
#include <algorithm>

//SGBM变量初始化
static cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create();

//图像读取
int imgRead(std::string path_left, std::string path_right);

//极线校正参数初始化
void Polar_Init();

//极线校正计算
void Polar_Compute();

//SGBM回调函数
void SGBM_CallBcck(int pos, void* userdata);

//SGBM立体匹配
void SGBM_Match();

//图像拼接
void Img_Merge(int gap);

//空洞填充
void insertDepth32f(cv::Mat& depth);

//视差图转点云图
void disp2cl();

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

//深度图
static cv::Mat depth;


//···············································································//
//SGBM系列参数

// 最小视差值
static int minDisparity = 0;

//水平sobel预处理后，映射滤波器大小
static int preFilterCap = 230;

// 匹配块大小，大于1的奇数
static int blockSize_ = 4; 

//视察搜索范围
static int numDisparities = 1;

// 平滑视差区域的最大尺寸，以考虑其噪声斑点和无效。将其设置为0可禁用斑点过滤。否则，将其设置在50 - 200的范围内。
static int speckleWindowSize = 200; 

// 视差变化阈值，每个连接组件内的最大视差变化。如果你做斑点过滤，将参数设置为正值，它将被隐式乘以16.通常，1或2就足够好了
static int speckleRange = 5; 

// 视差唯一性百分比， 视差窗口范围内最低代价是次低代价的(1 + uniquenessRatio / 100)倍时，最低代价对应的视差值才是该像素点的视差，否则该像素点的视差为 0，通常为5~15.
static int uniquenessRatio = 0; 

// 左右视差图的最大容许差异（超过将被清零），默认为 - 1，即不执行左右视差检查。
static int disp12MaxDiff = 1; 

// P1, P2控制视差图的光滑度
// 惩罚系数，一般：P1 = 8 * 通道数*SADWindowSize*SADWindowSize，P2 = 4 * P1
static int P1 = 1500;  
// p1控制视差平滑度，p2值越大，差异越平滑
static int P2 = 4 * 1500; 

//图片通道数
static int cn = img_left.channels();

