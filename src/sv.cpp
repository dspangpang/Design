#include "sv.h"

int imgRead(std::string path_left, std::string path_right){
	
    img_left_color = cv::imread(path_left, cv::IMREAD_UNCHANGED);
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
	P2 = 4 * P1;
	sgbm->setP2(P2);

	sgbm->setMode(cv::StereoSGBM::MODE_SGBM);

	sgbm->compute(img_left, img_right, disp);//输入图像必须为灰度图

	disp.convertTo(disp, CV_32F, 1.0 / 16); //低四位是小数部分，去除

	disp8U = cv::Mat(disp.rows, disp.cols, CV_8UC1);
	disp.convertTo(disp8U, CV_8U, 255/(numDisparities*16.));
    
	
	cv::imshow("Sgbm_Option", disp8U);
	cv::imwrite("../../img/disp.png", disp8U);
	
}

void SGBM_Match(){

	cv::namedWindow("Sgbm_Option", cv::WINDOW_FREERATIO);
		//注意类成员函数变为回调函数的方式
	cv::createTrackbar("preFilterCap", "Sgbm_Option", &preFilterCap, 255, SGBM_CallBcck);
	cv::createTrackbar("blockSize", "Sgbm_Option", &blockSize_, 21, SGBM_CallBcck);
	cv::createTrackbar("numDisparities", "Sgbm_Option", &numDisparities, 20, SGBM_CallBcck);
	cv::createTrackbar("speckleWindowSize", "Sgbm_Option", &speckleWindowSize, 500, SGBM_CallBcck);
	cv::createTrackbar("speckleRange", "Sgbm_Option", &speckleRange, 50, SGBM_CallBcck);
	cv::createTrackbar("uniquenessRatio", "Sgbm_Option", &uniquenessRatio, 20, SGBM_CallBcck);
	cv::createTrackbar("disp12MaxDiff", "Sgbm_Option", &disp12MaxDiff, 21, SGBM_CallBcck);
	cv::createTrackbar("P1", "Sgbm_Option", &P1, 5000, SGBM_CallBcck);

	SGBM_CallBcck(0, 0);
		
}

void insertDepth32f(cv::Mat& depth)
{
    const int width = depth.cols;
    const int height = depth.rows;
    float* data = (float*)depth.data;
    cv::Mat integralMap = cv::Mat::zeros(height, width, CV_64F);
    cv::Mat ptsMap = cv::Mat::zeros(height, width, CV_32S);
    double* integral = (double*)integralMap.data;
    int* ptsIntegral = (int*)ptsMap.data;
    memset(integral, 0, sizeof(double) * width * height);
    memset(ptsIntegral, 0, sizeof(int) * width * height);
    for (int i = 0; i < height; ++i)
    {
        int id1 = i * width;
        for (int j = 0; j < width; ++j)
        {
            int id2 = id1 + j;
            if (data[id2] > 1e-3)
            {
                integral[id2] = data[id2];
                ptsIntegral[id2] = 1;
            }
        }
    }
    // 积分区间
    for (int i = 0; i < height; ++i)
    {
        int id1 = i * width;
        for (int j = 1; j < width; ++j)
        {
            int id2 = id1 + j;
            integral[id2] += integral[id2 - 1];
            ptsIntegral[id2] += ptsIntegral[id2 - 1];
        }
    }
    for (int i = 1; i < height; ++i)
    {
        int id1 = i * width;
        for (int j = 0; j < width; ++j)
        {
            int id2 = id1 + j;
            integral[id2] += integral[id2 - width];
            ptsIntegral[id2] += ptsIntegral[id2 - width];
        }
    }
    int wnd;
    double dWnd = 2;
    while (dWnd > 1)
    {
        wnd = int(dWnd);
        dWnd /= 2;
        for (int i = 0; i < height; ++i)
        {
            int id1 = i * width;
            for (int j = 0; j < width; ++j)
            {
                int id2 = id1 + j;
                int left = j - wnd - 1;
                int right = j + wnd;
                int top = i - wnd - 1;
                int bot = i + wnd;
                left = std::max(0, left);
                right = std::min(right, width - 1);
                top = std::max(0, top);
                bot = std::min(bot, height - 1);
                int dx = right - left;
                int dy = (bot - top) * width;
                int idLeftTop = top * width + left;
                int idRightTop = idLeftTop + dx;
                int idLeftBot = idLeftTop + dy;
                int idRightBot = idLeftBot + dx;
                int ptsCnt = ptsIntegral[idRightBot] + ptsIntegral[idLeftTop] - (ptsIntegral[idLeftBot] + ptsIntegral[idRightTop]);
                double sumGray = integral[idRightBot] + integral[idLeftTop] - (integral[idLeftBot] + integral[idRightTop]);
                if (ptsCnt <= 0)
                {
                    continue;
                }
                data[id2] = float(sumGray / ptsCnt);
            }
        }
        int s = wnd / 2 * 2 + 1;
        if (s > 201)
        {
            s = 201;
        }
        cv::GaussianBlur(depth, depth, cv::Size(s, s), s, s);
    }
}


void disp2depth(){

	const double camera_fx = 1758.23;

	const double baseline = 88.39;

	depth = cv::Mat(img_left.rows, img_left.cols, CV_16UC1);

	for (int row = 0; row < depth.rows; row++)
    {
        for (int col = 0; col < depth.cols; col++)
        {
            ushort d = disp8U.ptr<uchar>(row)[col];

            if (d == 0)
                continue;

            depth.ptr<ushort>(row)[col] = camera_fx * baseline / d;
        }
    }

    cv::namedWindow("depth", cv::WINDOW_FREERATIO);
    cv::imshow("depth", depth);
    cv::imwrite("../../img/depth.png", depth);
}


void disp2depth_method(){

    cv::Mat Qx;
    Qx = (cv::Mat_<double>(4, 4) << 1., 0., 0., -977.42, 
                                    0., 1., 0., -552.15, 
                                    0., 0., 0., 1758.23,
                                    1., 0., -88.39, 0.);
    
    cv::reprojectImageTo3D(disp8U, Image3D , Qx);
    depth = cv::Mat(img_left.rows, img_left.cols, CV_32F);

	for (int row = 0; row < depth.rows; row++)
    {
        for (int col = 0; col < depth.cols; col++)
        {

            depth.ptr<float>(row)[col] = Image3D.ptr<float>(row)[col * 3 + 2];
        }
    }

    cv::namedWindow("depth", cv::WINDOW_FREERATIO);
    cv::imshow("depth", depth);

}

void getCloud(){

	const double camera_cx = 977.42;
	const double camera_cy = 552.15;
	const double camera_fx = 1758.23;
	const double camera_fy = 1758.23;

    const double camera_factor = 1000;             //单位换算因子

    cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());

      // 遍历深度图
    for (int m = 0; m < depth.rows; m++){

        for( int n = 0; n < depth.cols; n++){
            //获取深度图中(m,n)处的值
            //cv::Mat的ptr函数会返回指向该图像第m行数据的头指针。然后加上位移n后，这个指针指向的数据就是我们需要读取的数据.
            ushort d = depth.ptr<ushort>(m)[n];
            // d 可能没有值，若如此，跳过此点
            if (d == 0)
                continue;
            //d 存在值，则向点云增加一个点
            pcl::PointXYZRGB p;
            
            // 计算这个点的空间坐标
            p.z = Image3D.ptr<float>(m)[n * 3 + 2];
            p.x = Image3D.ptr<float>(m)[n * 3];
            p.y = Image3D.ptr<float>(m)[n * 3 + 1];

		    p.b = img_left_color.ptr<uchar>(m)[n * 3];
		    p.g = img_left_color.ptr<uchar>(m)[n * 3 + 1];
		    p.r = img_left_color.ptr<uchar>(m)[n * 3 + 2];
            
            //把p加入到点云中
            cloud->points.push_back(p); 
        }
    }

    cloud->height = 1;
    cloud->width = cloud->points.size();
    cloud->is_dense = false;
    pcl::io::savePCDFile( "../../img/pointcloud.pcd", *cloud);

    pcl::visualization::CloudViewer viewer("viewer");   
    viewer.showCloud(cloud);
    while (!viewer.wasStopped())
    {

    }  

}