#include "types.h"
#include "pcl_lib.h"

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
    std::string path_left = argc[1];
    std::string path_right = argc[2];

    cv::Mat img_left_c = cv::imread(path_left, cv::IMREAD_COLOR);
    cv::Mat img_left = cv::imread(path_left, cv::IMREAD_GRAYSCALE);
    cv::Mat img_right = cv::imread(path_right, cv::IMREAD_GRAYSCALE);

    if (img_left.data == nullptr || img_right.data == nullptr) {
        std::cout << "读取影像失败！" << std::endl;
        return -1;
    }
    if (img_left.rows != img_right.rows || img_left.cols != img_right.cols) {
        std::cout << "左右影像尺寸不一致！" << std::endl;
        return -1;
    }

    //···············································································//
    //输入初始参数

    Polar_Option Option1;

    cv::Mat left_out;
    cv::Mat right_out;
    //内参矩阵
    Option1.cameraMatrixL = (cv::Mat_<double>(3, 3) << 7317.27331, 0., 1249.79388, 0., 7329.17800, 1013.03051, 0., 0., 1.);
    Option1.cameraMatrixR = (cv::Mat_<double>(3, 3) << 7305.26384, 0., 1202.77811, 0., 7321.69551, 1007.91219, 0., 0., 1.);

    //畸变参数
    Option1.distCoeffL = (cv::Mat_<double>(1, 5) << 0.11689, 0.91004, 0, 0, 0);
    Option1.distCoeffR = (cv::Mat_<double>(1, 5) << 0.03394, 1.25432, 0, 0, 0);

    //左右坐标系转换关系
    Option1.R2L_R = (cv::Mat_<double>(3, 3)  << 0.9410, 0.0030, 0.3383,
                                              -0.0010, 0.9999, -0.0015,
                                              -0.3384, 0.0112, 0.9410);

    Option1.R2L_T = (cv::Mat_<double>(3, 1) << -624.33394, 1.47164, 73.85077);

    Option1.image_size = cv::Size(398, 333);

    //···············································································//
    //极线校正

    cv::stereoRectify(Option1.cameraMatrixL, Option1.distCoeffL, Option1.cameraMatrixR, Option1.distCoeffR, 
                        Option1.image_size, Option1.R2L_R, Option1.R2L_T, Option1.RL, Option1.RR, Option1.PL, Option1.PR,
                        Option1.Q, cv::CALIB_ZERO_DISPARITY);

    cv::initUndistortRectifyMap(Option1.cameraMatrixL, Option1.distCoeffL, Option1.RL, Option1.PL, 
                                    Option1.image_size, CV_16SC2, Option1.rmap[0][0], Option1.rmap[0][1]);
    cv::initUndistortRectifyMap(Option1.cameraMatrixR, Option1.distCoeffR, Option1.RR, Option1.PR, 
                                    Option1.image_size, CV_16SC2, Option1.rmap[1][0], Option1.rmap[1][1]);
    
    cv::remap(img_left, left_out, Option1.rmap[0][0], Option1.rmap[0][1], cv::INTER_AREA);//左校正
    cv::remap(img_right, right_out, Option1.rmap[1][0], Option1.rmap[1][1], cv::INTER_AREA);//右校正

    cv::Mat merge;
	cv::hconcat(img_left, img_right, merge);

    for( int i = 0; i < Option1.image_size.height; i += 16)
    {
        cv::line(merge, cv::Point(0, i), cv::Point(2 * Option1.image_size.width, i), cv::Scalar(255), 1, 8);
    }

    cv::imshow("结果",merge);
    cv::imwrite("../../img/merge.jpg",merge);

    
    //···············································································//

    cv::waitKey(0);
    return 0;
}