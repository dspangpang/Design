#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

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
