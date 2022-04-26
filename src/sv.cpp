#include "sv.h"

int imgRead(std::string path_left, std::string path_right){
	
	img_left = cv::imread(path_left, cv::IMREAD_GRAYSCALE);
    img_right = cv::imread(path_right, cv::IMREAD_GRAYSCALE);

    if (img_left.data == nullptr || img_right.data == nullptr) {
        std::cout << "读取影像失败！" << std::endl;
        return -1;
    }
    if (img_left.rows != img_right.rows || img_left.cols != img_right.cols) {
        std::cout << "左右影像尺寸不一致！" << std::endl;
        return -1;
    }

}

void Polar_Init(){

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

    Option1.image_size = cv::Size(img_left.cols, img_left.rows);
}


void Polar_Compute(){

    Polar_Init();

    cv::stereoRectify(Option1.cameraMatrixL, Option1.distCoeffL, Option1.cameraMatrixR, Option1.distCoeffR, 
                        Option1.image_size, Option1.R2L_R, Option1.R2L_T, Option1.RL, Option1.RR, Option1.PL, Option1.PR,
                        Option1.Q, cv::CALIB_ZERO_DISPARITY);

    cv::initUndistortRectifyMap(Option1.cameraMatrixL, Option1.distCoeffL, Option1.RL, Option1.PL, 
                                    Option1.image_size, CV_16SC2, Option1.rmap[0][0], Option1.rmap[0][1]);
    cv::initUndistortRectifyMap(Option1.cameraMatrixR, Option1.distCoeffR, Option1.RR, Option1.PR, 
                                    Option1.image_size, CV_16SC2, Option1.rmap[1][0], Option1.rmap[1][1]);
    
    cv::remap(img_left, left_out, Option1.rmap[0][0], Option1.rmap[0][1], cv::INTER_AREA);//左校正
    cv::remap(img_right, right_out, Option1.rmap[1][0], Option1.rmap[1][1], cv::INTER_AREA);//右校正
}


void Img_Merge(int gap){

    cv::hconcat(img_left, img_right, merge);

    for( int i = 0; i < Option1.image_size.height; i += gap)
    {
        cv::line(merge, cv::Point(0, i), cv::Point(2 * Option1.image_size.width, i), cv::Scalar(255), 1, 8);
    }

    cv::imwrite("../../img/merge.jpg",merge);
}


void SGBM_Match(int pos, void* userdata){

	int cn = img_left.channels();

	//设定SAD计算窗口的大小，窗口大小应为奇数，最小值为5
	int blockSize = cv::getTrackbarPos("blockSize", "SGBM_disparity");
	if (blockSize % 2 == 0){
		blockSize += 1;
	}
	if (blockSize < 5){
		blockSize = 5;
	}
	sgbm->setBlockSize(blockSize);

	//视差搜索的起始位置
	sgbm->setMinDisparity(0);
	//视差搜索范围，其值必须为16的整数倍
	sgbm->setNumDisparities(cv::getTrackbarPos("numDisparities", "SGBM_disparity"));

	sgbm->setSpeckleWindowSize(cv::getTrackbarPos("speckleWindowSize", "SGBM_disparity"));
	sgbm->setSpeckleRange(cv::getTrackbarPos("speckleRange", "SGBM_disparity"));
	sgbm->setUniquenessRatio(cv::getTrackbarPos("uniquenessRatio", "SGBM_disparity"));
	sgbm->setDisp12MaxDiff(cv::getTrackbarPos("disp12MaxDiff", "SGBM_disparity"));

	/*int P1 = 8 * left.channels() * SADWindowSize* SADWindowSize;
	int P2 = 32 * left.channels() * SADWindowSize* SADWindowSize;*/
	// 惩罚系数，一般：P1=8*通道数*SADWindowSize*SADWindowSize，P2=4*P1
	int p1 = 8 * cn * blockSize * blockSize;
	sgbm->setP1(p1);
	// p1控制视差平滑度，p2值越大，差异越平滑
	sgbm->setP2(4 * p1);

	sgbm->setMode(cv::StereoSGBM::MODE_SGBM);

	sgbm->compute(img_left, img_right, disp);//输入图像必须为灰度图

	cv::imshow("disp", disp);
	cv::imwrite("../../img/disp.jpg", disp);
}

void SGBM_Window(){

	// 最小视差值
	int minDisparity = 0;

	// 视差范围，即最大视差值和最小视差值之差，必须是16的倍数。
	int numDisparities = 64;

	// 匹配块大小，大于1的奇数
	int blockSize = 5; 

	// P1, P2控制视差图的光滑度
	// 惩罚系数，一般：P1 = 8 * 通道数*SADWindowSize*SADWindowSize，P2 = 4 * P1
	int P1 = 600;  
	// p1控制视差平滑度，p2值越大，差异越平滑
	int P2 = 2400; 

	// 左右视差图的最大容许差异（超过将被清零），默认为 - 1，即不执行左右视差检查。
	int disp12MaxDiff = 200; 

	//水平sobel预处理后，映射滤波器大小
	int preFilterCap = 0;

	// 视差唯一性百分比， 视差窗口范围内最低代价是次低代价的(1 + uniquenessRatio / 100)倍时，最低代价对应的视差值才是该像素点的视差，否则该像素点的视差为 0，通常为5~15.
	int uniquenessRatio = 6; 

	// 平滑视差区域的最大尺寸，以考虑其噪声斑点和无效。将其设置为0可禁用斑点过滤。否则，将其设置在50 - 200的范围内。
	int speckleWindowSize = 60; 
	// 视差变化阈值，每个连接组件内的最大视差变化。如果你做斑点过滤，将参数设置为正值，它将被隐式乘以16.通常，1或2就足够好了
	int speckleRange = 2; 

	cv::namedWindow("Sgbm_Option", 1);

	//注意类成员函数变为回调函数的方式
	cv::createTrackbar("blockSize", "Sgbm_Option", NULL, 21, SGBM_Match);
	cv::createTrackbar("numDisparities", "Sgbm_Option", NULL, 256, SGBM_Match);
	cv::createTrackbar("speckleWindowSize", "Sgbm_Option", NULL, 200, SGBM_Match);
	cv::createTrackbar("speckleRange", "Sgbm_Option", NULL, 2, SGBM_Match);
	cv::createTrackbar("uniquenessRatio", "Sgbm_Option", NULL, 20, SGBM_Match);
	cv::createTrackbar("disp12MaxDiff", "Sgbm_Option", NULL, 21, SGBM_Match);

}

