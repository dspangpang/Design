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

	return 0;

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


void SGBM_CallBcck(int pos, void* userdata){

	//设定SAD计算窗口的大小，窗口大小应为奇数，最小值为5
	int blockSize = blockSize_;
	if (blockSize % 2 == 0){
		blockSize += 1;
	}
	if (blockSize < 3){
		blockSize = 3;
	}
	sgbm->setBlockSize(blockSize);

	//视差搜索的起始位置
	sgbm->setMinDisparity(0);
	//视差搜索范围，其值必须为16的整数倍
	if (numDisparities < 1){
		numDisparities = 1;
	}
	sgbm->setNumDisparities(numDisparities * 16);

	//视差搜索范围，其值必须为16的整数倍
	sgbm->setPreFilterCap(preFilterCap);

	sgbm->setSpeckleWindowSize(speckleWindowSize);
	sgbm->setSpeckleRange(speckleRange);
	sgbm->setUniquenessRatio(uniquenessRatio);
	sgbm->setDisp12MaxDiff(disp12MaxDiff);

	/*int P1 = 8 * left.channels() * SADWindowSize* SADWindowSize;
	int P2 = 32 * left.channels() * SADWindowSize* SADWindowSize;*/
	// 惩罚系数，一般：P1=8*通道数*SADWindowSize*SADWindowSize，P2=4*P1
//	int p1 = 8 * cn * blockSize * blockSize;
	sgbm->setP1(P1);
	// p1控制视差平滑度，p2值越大，差异越平滑
	if(P2 < P1){
		P2 = P1 + 50;
	}
	sgbm->setP2(P2);

	sgbm->setMode(cv::StereoSGBM::MODE_SGBM);

	sgbm->compute(img_left, img_right, disp);//输入图像必须为灰度图

	disp.convertTo(disp, CV_32F, 1.0 / 16);

	cv::Mat disp8U = cv::Mat(disp.rows, disp.cols, CV_8UC1);

	disp.convertTo(disp8U,CV_8U,255/(numDisparities*16.));

	cv::imshow("Sgbm_Option", disp8U);
	cv::imwrite("../../img/disp.jpg", disp8U);
	
}

void SGBM_Match(){

	cv::namedWindow("Sgbm_Option", cv::WINDOW_AUTOSIZE);
		//注意类成员函数变为回调函数的方式
	cv::createTrackbar("preFilterCap", "Sgbm_Option", &preFilterCap, 100, SGBM_CallBcck);
	cv::createTrackbar("blockSize", "Sgbm_Option", &blockSize_, 21, SGBM_CallBcck);
	cv::createTrackbar("numDisparities", "Sgbm_Option", &numDisparities, 20, SGBM_CallBcck);
	cv::createTrackbar("speckleWindowSize", "Sgbm_Option", &speckleWindowSize, 200, SGBM_CallBcck);
	cv::createTrackbar("speckleRange", "Sgbm_Option", &speckleRange, 10, SGBM_CallBcck);
	cv::createTrackbar("uniquenessRatio", "Sgbm_Option", &uniquenessRatio, 20, SGBM_CallBcck);
	cv::createTrackbar("disp12MaxDiff", "Sgbm_Option", &disp12MaxDiff, 21, SGBM_CallBcck);
	cv::createTrackbar("P1", "Sgbm_Option", &P1, 1000, SGBM_CallBcck);
	cv::createTrackbar("P2", "Sgbm_Option", &P2, 4000, SGBM_CallBcck);
	SGBM_CallBcck(0, 0);
		
}

