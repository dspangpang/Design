#include "sv.h"
#include <iostream>

/**
 * \brief
 * \param argv 3
 * \param argc argc[1]:左影像路径 argc[2]: 右影像路径 
 * \return
 */
int main(int argv, char** argc){
    if (argv < 3) {
        std::cout << "参数过少，请至少指定左右影像路径！" << std::endl;
        return -1;
    }
    //···············································································//
    // 读取影像
    int err ;
    err = imgRead(argc[1], argc[2]);
    if(err < 0){
        std::cout << "程序结束" << std::endl;
        return -1;
    }
    std::cout << "图像读取成功" << std::endl;
    //···············································································//
    //极线校正

    //···············································································//
    //图像拼接

    //···············································································//
    //SGBM视差计算

    SGBM_Match();

    //···············································································//
    //获取点云数据
    disp2depth_method();
    getCloud();
    //getMesh();

    cv::waitKey();
    cv::destroyAllWindows();
    
    return 0;
}