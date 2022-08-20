/*****************************************************************************
**       PL-SVO: Semi-direct Monocular Visual Odometry by Combining       	**
**                        Points and Line Segments                          **
******************************************************************************
**																			**
**	Copyright(c) 2016, Ruben Gomez-Ojeda, University of Malaga              **
**	Copyright(c) 2016, MAPIR group, University of Malaga					**
**																			**
**  This library extends SVO to the case of also using line-segments, thus  **
**  it is highly based on the previous implementation of SVO:               **
**  https://github.com/uzh-rpg/rpg_svo                                      **
**                                                                          **
**  This program is free software: you can redistribute it and/or modify	**
**  it under the terms of the GNU General Public License (version 3) as		**
**	published by the Free Software Foundation.								**
**																			**
**  This program is distributed in the hope that it will be useful, but		**
**	WITHOUT ANY WARRANTY; without even the implied warranty of				**
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the			**
**  GNU General Public License for more details.							**
**																			**
**  You should have received a copy of the GNU General Public License		**
**  along with this program.  If not, see <http://www.gnu.org/licenses/>.	**
**																			**
*****************************************************************************/

#ifdef HAS_MRPT
#include <plsvo/sceneRepresentation.h>
#endif
#include <boost/filesystem.hpp>
#include <plsvo/config.h>
#include <plsvo/frame_handler_mono.h>
#include <plsvo/map.h>
#include <plsvo/feature_detection.h>
#include <plsvo/depth_filter.h>
#include <plsvo/frame.h>
#include <plsvo/feature.h>
#include <plsvo/feature3D.h>
#include <plsvo/point.h>
#include <vector>
#include <string>
#include <vikit/math_utils.h>
#include <vikit/vision.h>
#include <vikit/abstract_camera.h>
#include <vikit/atan_camera.h>
#include <vikit/pinhole_camera.h>
#include <opencv2/opencv.hpp>
#include <sophus/se3.h>
#include <iostream>
#include "opencv2/core/utility.hpp"
#include "yaml-cpp/yaml.h"

#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/at_c.hpp>

using namespace std;
Eigen::Quaternion<double, Eigen::DontAlign> varName;

string files="";
string calib="";
string calib_path="";
cv::Mat undistortFishEye(const cv::Mat &distorted, const float w);

// 首先定义了 svo_options的数据结构，里面包含的是程序的运行参数(一堆参数选项)
struct svo_options {
    int seq_offset;
    int seq_step;
    int seq_length;
    bool has_points;
    bool has_lines;
    bool is_tum;
    string dataset_dir;
    string images_dir;
    string traj_out;
    string map_out;
};

// 在plsvo命名空间下声明了一个 ConvergedSeed 结构类型和BenchmarkNode类
namespace plsvo {

struct ConvergedSeed { //汇聚结构
  int x_, y_;
  Vector3d pos_;
  cv::Vec3b col_;
  ConvergedSeed(int x, int y, Vector3d pos, cv::Vec3b col) :
    x_(x), y_(y), pos_(pos), col_(col)
  {}
};

class BenchmarkNode// 基准类 //详细介绍BenchmarkNode类，他包含了主程序的主要功能函数
{//4个私有成员
  vk::AbstractCamera* cam_;
  FrameHandlerMono* vo_;
  DepthFilter* depth_filter_;
  std::list<ConvergedSeed> results_;

public://公开的成员函数
  BenchmarkNode(vk::AbstractCamera *cam_);
  BenchmarkNode(vk::AbstractCamera *cam_, const plsvo::FrameHandlerMono::Options& handler_opts);
  ~BenchmarkNode();
  void depthFilterCbPt(plsvo::Point* point);
  void depthFilterCbLs(plsvo::LineSeg* ls);
  int runFromFolder(svo_options opts);
  int runFromFolder(vk::PinholeCamera* cam_, svo_options opts);
  int runFromFolder(vk::ATANCamera* cam_,    svo_options opts);
};

//负责启动视觉里程计的同名构造函数
BenchmarkNode::BenchmarkNode(vk::AbstractCamera* cam_)//同名构造函数1   参数不同
{
  vo_ = new plsvo::FrameHandlerMono(cam_);
  vo_->start();
}

BenchmarkNode::BenchmarkNode(
    vk::AbstractCamera* cam_,
    const plsvo::FrameHandlerMono::Options& handler_opts)//同名构造函数2    参数不同
{
  vo_ = new plsvo::FrameHandlerMono(cam_, handler_opts);
  vo_->start();
}

//一个类有且仅有一个析构函数。如果定义类时没写析构函数，则编译器生成默认析构函数。如果定义了析构函数，则编译器不生成默认析构函数。
BenchmarkNode::~BenchmarkNode()//析构函数
{
  delete vo_;
  delete cam_;
}

//点和线特征的深度滤波器函数
void BenchmarkNode::depthFilterCbPt(plsvo::Point* point)//point点的深度滤波器
{
  cv::Vec3b color = point->obs_.front()->frame->img_pyr_[0].at<cv::Vec3b>(point->obs_.front()->px[0], point->obs_.front()->px[1]);
  results_.push_back(ConvergedSeed(
      point->obs_.front()->px[0], point->obs_.front()->px[1], point->pos_, color));
  delete point->obs_.front();
}

void BenchmarkNode::depthFilterCbLs(plsvo::LineSeg* ls)//line线的深度滤波器
{
  cv::Vec3b color = ls->obs_.front()->frame->img_pyr_[0].at<cv::Vec3b>(ls->obs_.front()->spx[0], ls->obs_.front()->spx[1]);
  results_.push_back(ConvergedSeed(
      ls->obs_.front()->spx[0], ls->obs_.front()->spx[1], ls->spos_, color)); // test only with spoint
  delete ls->obs_.front();
}

//int BenchmarkNode::runFromFolder是算法的主流程，读取数据文件并运行。
int BenchmarkNode::runFromFolder(svo_options opts)
{
    // grab options
    int seq_offset     = opts.seq_offset;
    int seq_step       = opts.seq_step;
    int seq_length     = opts.seq_length;
    bool has_points    = opts.has_points;
    bool has_lines     = opts.has_lines;
    bool is_tum        = opts.is_tum;
    string dataset_dir = opts.dataset_dir;
    string images_dir  = opts.images_dir;
    string map_out     = opts.map_out;
    string traj_out    = opts.traj_out;
    int fps_           = 30;

    // Read content of the .yaml dataset configuration file //读参数文件
    YAML::Node dset_config = YAML::LoadFile(dataset_dir+"/dataset_params.yaml");

    // get a sorted list of files in the img directory //读img
    boost::filesystem::path img_dir_path(images_dir.c_str());
    if (!boost::filesystem::exists(img_dir_path))
    {
        cout << endl << "Image directory does not exist: \t" << images_dir << endl;
        return -1;
    }

    // get all files in the img directory
    size_t max_len = 0;
    std::list<std::string> imgs;
    boost::filesystem::directory_iterator end_itr; // default construction yields past-the-end
    for (boost::filesystem::directory_iterator file(img_dir_path); file != end_itr; ++file)
    {
        boost::filesystem::path filename_path = file->path().filename();
        if (boost::filesystem::is_regular_file(file->status()) &&
                (filename_path.extension() == ".png"  ||
                 filename_path.extension() == ".jpg"  ||
                 filename_path.extension() == ".jpeg" ||
                 filename_path.extension() == ".tiff") )
        {
            std::string filename(filename_path.string());
            imgs.push_back(filename);
            max_len = std::max(max_len, filename.length());
        }
    }

    // sort them by filename; add leading zeros to make filename-lengths equal if needed
    // 按文件名排序; 如果需要，添加前导零以使文件名长度相等
    std::map<std::string, std::string> sorted_imgs;
    int n_imgs = 0;
    for (std::list<std::string>::iterator img = imgs.begin(); img != imgs.end(); ++img)
    {
        sorted_imgs[std::string(max_len - img->length(), '0') + (*img)] = *img;
        n_imgs++;
    }

    // add offset / step / length
    int seq_end = n_imgs;
    if( seq_length != 0 )
        int seq_end = std::min( seq_length*seq_step+seq_offset, n_imgs );
    std::map<std::string, std::string> sorted_imgs_aux = sorted_imgs;//根据初始偏移，步长和数据长度把图像数据存储到
    sorted_imgs.clear();
    int k = 0;
    for (auto img = sorted_imgs_aux.begin(); img != sorted_imgs_aux.end(); std::advance(img,seq_step) )
    {
        if( k >= seq_offset && k <= seq_end )
            sorted_imgs.insert( *img );
        k++;
    }

    // create scene//场景初始化
    sceneRepresentation scene("../app/scene_config.ini");
    Matrix<double,4,4> T_c_w, T_f_w = Matrix<double,4,4>::Identity(), T_f_w_prev = Matrix<double,4,4>::Identity(), T_inc;
    T_c_w = Matrix<double,4,4>::Identity();
    scene.initializeScene(T_f_w);

    // run SVO (pose estimation)
    std::list<FramePtr> frames;
    int frame_counter = 1;
    std::ofstream ofs_traj(traj_out.c_str());
    for (std::map<std::string, std::string>::iterator it = sorted_imgs.begin(); it != sorted_imgs.end(); ++it)
    {
        // load image
        boost::filesystem::path img_path = img_dir_path / boost::filesystem::path(it->second.c_str());
        if (frame_counter == 1)
            std::cout << "reading image " << img_path.string() << std::endl;
        cv::Mat img(cv::imread(img_path.string(), CV_8UC3));//进入循环读入图像进行处理
        //cv::cvtColor( img, img, cv::COLOR_BGR2GRAY);
        // IMPORTANT: The image must be flipped if focal length is negative
        // since the optimization code assumes that both f_x and f_y are positive
        {
          // get camera config
          YAML::Node cam_config = dset_config["cam0"];
          // check f_y sign
          if(cam_config["cam_fy"].as<double>()<0)
            cv::flip(img, img, 0); // Vertical flipping (around x axis, changes y coordinate)
        }
        assert(!img.empty());

        // process frame
        vo_->addImage(img, frame_counter / (double)fps_);

        // display tracking quality
        if (vo_->lastFrame() != NULL) {
            std::cout << "Frame-Id: "       << vo_->lastFrame()->id_ << " \t"
                      << "#PointFeatures: " << vo_->lastNumPtObservations() << " \t"
                      << "#LineFeatures: "  << vo_->lastNumLsObservations() << " \t"
                      << "Proc. Time: "     << vo_->lastProcessingTime()*1000 << "ms" << std::endl << std::endl;
            frames.push_back(vo_->lastFrame());
            // access the pose of the camera via vo_->lastFrame()->T_f_w_.
            T_f_w = vo_->lastFrame()->T_f_w_.matrix();
            T_inc = T_f_w_prev * T_f_w.inverse();
            T_f_w_prev = T_f_w;
            scene.setPose( T_inc );
            scene.setText( vo_->lastFrame()->id_, vo_->lastProcessingTime()*1000, vo_->lastNumPtObservations(), 0, vo_->lastNumLsObservations(), 0 );
            cv::Mat dbg_img = vo_->get_debug_image();
            if( !dbg_img.empty() )
                scene.setImage( dbg_img );
            else
                scene.setImage("./empty_img.png");

            // save trajectory
            Eigen::Matrix<double, 6, 6> cov = vo_->lastFrame()->Cov_;
            bool skip_frame = false;
            for (int i = 0; i < 6; i++)
                for (int j = 0; j < 6; j++)
                    if (! ((1.e-16 < fabs(cov(i,j))) && (fabs(cov(i,j)) < 1.e+16)) )    // likely an invalid pose -> seriously?
                        skip_frame = true;

            // access the pose of the camera via vo_->lastFrame()->T_f_w_.
            Sophus::SE3 world_transf = vo_->lastFrame()->T_f_w_.inverse();
            Eigen::Quaterniond quat  = world_transf.unit_quaternion();
            Eigen::Vector3d transl   = world_transf.translation();
            if ( ((transl(0) == 0.) && (transl(1) == 0.) && (transl(2) == 0.)) &&
                    ((quat.x() == -0.) && (quat.y() == -0.) && (quat.z() == -0.) && (quat.w() == 1.)) )
                skip_frame = true;

            if (skip_frame)
                continue;
                //cout << endl << "Frame skipped, watch out" << endl;

            string filename  = (it->second).c_str();
            int    size = filename.size();
            string timestamp =  filename.substr(0,size-4);
            ofs_traj  << timestamp << " "
                      << transl(0) << " " << transl(1) << " " << transl(2) << " "
                      << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w()
                      << std::endl;

            // introduce 3d features to the scene
            vector< Matrix<double,3,1> > points3d;
            for(list<PointFeat*>::iterator it=vo_->lastFrame()->pt_fts_.begin(); it!=vo_->lastFrame()->pt_fts_.end();++it)
                if((*it)->feat3D != NULL)
                   points3d.push_back( (*it)->feat3D->pos_ );
            scene.setPointsSVO(points3d);
            // introduce 3d line segments to the scene
            vector< Matrix<double,6,1> > lines3d;
            Matrix<double,6,1> line3d_;
            for(list<LineFeat*>::iterator it=vo_->lastFrame()->seg_fts_.begin(); it!=vo_->lastFrame()->seg_fts_.end(); ++it)
            {
              if((*it)->feat3D != NULL)
              {
                line3d_ << (*it)->feat3D->spos_, (*it)->feat3D->epos_;
                lines3d.push_back( line3d_ );
              }
            }
            scene.setLinesSVO(lines3d);
            // update scene
            scene.updateScene();
        }
        frame_counter++;
    }
    cout << endl << "End of the SVO pipeline" << endl << endl;

    std::cout << "Done." << std::endl;
    return 0;
}

int BenchmarkNode::runFromFolder(vk::PinholeCamera* cam_, svo_options opts) // 主要程序
{
    std::cout<<"=========Pinhole camera model==========="<<std::endl;
    // grab options
    int seq_offset     = opts.seq_offset;
    int seq_step       = opts.seq_step;
    int seq_length     = opts.seq_length;
    bool has_points    = opts.has_points;
    bool has_lines     = opts.has_lines;
    bool is_tum        = opts.is_tum;
    string dataset_dir = opts.dataset_dir;
    string images_dir  = opts.images_dir;
    string map_out     = opts.map_out;
    string traj_out    = opts.traj_out;
    int fps_           = 30;

    // Read content of the .yaml dataset configuration file //读参数文件
    // YAML::Node dset_config = YAML::LoadFile(dataset_dir+"/dataset_params.yaml");
    YAML::Node dset_config = YAML::LoadFile(dataset_dir+calib);
    // YAML::Node dset_config = YAML::LoadFile("/home/wys/slam/data/EuRoC/MH_01_easy/mav0/cam0/dataset_params.yaml");

    // get a sorted list of files in the img directory //读img
    boost::filesystem::path img_dir_path(images_dir.c_str());
    if (!boost::filesystem::exists(img_dir_path))
    {
        cout << endl << "Image directory does not exist: \t" << images_dir << endl;
        return -1;
    }

    // get all files in the img directory
    size_t max_len = 0;
    //size_t 类型定义在cstddef头文件中，该文件是C标准库的头文件stddef.h的C++版。
    //它是一个与机器相关的unsigned类型，其大小足以保证存储内存中对象的大小。
    //在C++中，设计size_t 就是为了适应多个平台的
    std::list<std::string> imgs;
    boost::filesystem::directory_iterator end_itr; // default construction yields past-the-end
    for (boost::filesystem::directory_iterator file(img_dir_path); file != end_itr; ++file)//文件迭代器
    {
        boost::filesystem::path filename_path = file->path().filename();
        if (boost::filesystem::is_regular_file(file->status()) &&
                (filename_path.extension() == ".png"  ||
                 filename_path.extension() == ".jpg"  ||
                 filename_path.extension() == ".jpeg" ||
                 filename_path.extension() == ".tiff") )//img的格式检测
        {
            std::string filename(filename_path.string());
            imgs.push_back(filename);//读取img
            max_len = max(max_len, filename.length());//文件名长度？
            //printf("%d|%d|%s\n",max_len,filename.length(),filename.c_str());
        }
    }

    // sort them by filename; add leading zeros to make filename-lengths equal if needed
    // 按文件名排序; 如果需要，添加前导零以使文件名长度相等
    std::map<std::string, std::string> sorted_imgs;//容器map
    int n_imgs = 0;
    for (std::list<std::string>::iterator img = imgs.begin(); img != imgs.end(); ++img)//iterator迭代器 map双向迭代器
    {
        //string str=std::string(max_len - img->length(), '0');
        //std::cout<<str<<std::endl;
        // if(max_len - img->length()!=0){
        //     string str=std::string(max_len - img->length(), '0');
        //     std::cout<<str<<std::endl;
        // }
        //printf("%d|%d|%s\n",max_len,img->length(),str.c_str());
        sorted_imgs[std::string(max_len - img->length(), '0') + (*img)] = *img;//list容器imgs的迭代器指向的元素*
        // std::string(max_len - img->length(), '0') 在名字不一样长的img前面加0，补齐img长度一样长
        // std::cout<<*img<<std::endl;
        n_imgs++;
    }
    //用map(str,str) sorted_imgs存储了所有的图像

    // add offset / step / length //按照步长取一部分img
    int seq_end = n_imgs;
    if( seq_length != 0 )
        int seq_end = std::min( seq_length*seq_step+seq_offset, n_imgs );
    std::map<std::string, std::string> sorted_imgs_aux = sorted_imgs;//根据初始偏移，步长和数据长度把图像数据存储到
    sorted_imgs.clear();
    int k = 0;
    for (auto img = sorted_imgs_aux.begin(); img != sorted_imgs_aux.end(); std::advance(img,seq_step) )//迭代器函数advance,移动迭代器指向的元素
    {
        if( k >= seq_offset && k <= seq_end )//设定选取的img的范围
            sorted_imgs.insert( *img );
        k++;
    }

    // create scene//场景初始化         //运行SVO进行姿态估计
    // 读入mrpt显示配置
    sceneRepresentation scene("../app/scene_config.ini");
    //Identity()，用以初始一个单位阵，但这个函数只针对Matrix类型，不针对Array类型
    Matrix<double,4,4> T_c_w = Matrix<double,4,4>::Identity();
    Matrix<double,4,4> T_f_w = Matrix<double,4,4>::Identity();
    Matrix<double,4,4> T_f_w_prev = Matrix<double,4,4>::Identity();
    Matrix<double,4,4> T_inc;
    scene.initializeScene(T_f_w);

    // run SVO (pose estimation)//运行svo，位姿估计
    std::list<FramePtr> frames;
    int frame_counter = 1;
    std::ofstream ofs_traj(traj_out.c_str());//输出路径的txt文件
    for (std::map<std::string, std::string>::iterator it = sorted_imgs.begin(); it != sorted_imgs.end(); ++it )
    {
        // load image
        // // //test
        // // std::cout<<"=========="<<it->first.c_str()<<std::endl;
        // // std::cout<<"=========="<<it->second.c_str()<<std::endl;
        // // std::cout<<"=========="<<img_dir_path<<std::endl;
        // // std::cout<<"=========="<< img_dir_path / boost::filesystem::path(it->second.c_str())<<std::endl;
        // // boost::filesystem::path 创建路径path //boost c++拓展标准库
        //img_path为每张图像的绝对路径；img_dir_path图像的文件夹；it->second.c_str()为图像名
        boost::filesystem::path img_path = img_dir_path / boost::filesystem::path(it->second.c_str());
        if (frame_counter == 1)
            std::cout << "reading image " << img_path.string() << std::endl;
        cv::Mat img(cv::imread(img_path.string(), CV_8UC1));//进入循环读入图像进行处理
        // IMPORTANT: The image must be flipped if focal length is negative //焦距为负数，图像需要翻转
        // since the optimization code assumes that both f_x and f_y are positive //优化代码假设焦距为正数
        {
          // get camera config//获得相机参数
          YAML::Node cam_config = dset_config["cam0"];
          // check f_y sign 
          if(cam_config["cam_fy"].as<double>()<0)//.as<double>()：yaml把参数作为<double>读出来
            cv::flip(img, img, 0); // Vertical flipping (around x axis, changes y coordinate)//焦距为负数，图像需要翻转
        }
        assert(!img.empty());//保证img不空

        // undistort image
        cv::Mat img_rec;
        cam_->undistortImage(img,img_rec);//图像去畸变     //vikit库

        // process frame    //开始对图像进行处理
        vo_->addImage(img_rec, frame_counter / (double)fps_);// 实现在frame_handler_mono.cpp 自己写的

        // display tracking quality
        // 显示跟踪质量
        if (vo_->lastFrame() != NULL) {
            std::cout << "Frame-Id: "       << vo_->lastFrame()->id_ << " \t"
                      << "#PointFeatures: " << vo_->lastNumPtObservations() << " \t"
                      << "#LineFeatures: "  << vo_->lastNumLsObservations() << " \t"
                      << "Proc. Time: "     << vo_->lastProcessingTime()*1000 << "ms" << std::endl << std::endl;
            frames.push_back(vo_->lastFrame());

            // save trajectory
            Eigen::Matrix<double, 6, 6> cov = vo_->lastFrame()->Cov_;
            bool skip_frame = false;
            for (int i = 0; i < 6; i++)
                for (int j = 0; j < 6; j++)
                    if (! ((1.e-16 < fabs(cov(i,j))) && (fabs(cov(i,j)) < 1.e+16)) )    // likely an invalid pose -> seriously?
                        skip_frame = true;

            // access the pose of the camera via vo_->lastFrame()->T_f_w_.
            Sophus::SE3 world_transf = vo_->lastFrame()->T_f_w_.inverse();
            Eigen::Quaterniond quat  = world_transf.unit_quaternion();
            Eigen::Vector3d transl   = world_transf.translation();
            Matrix3d rot = world_transf.rotation_matrix();
            if ( ((transl(0) == 0.) && (transl(1) == 0.) && (transl(2) == 0.)) &&
                    ((quat.x() == -0.) && (quat.y() == -0.) && (quat.z() == -0.) && (quat.w() == 1.)) )
                skip_frame = true;

            if (skip_frame)
                continue;
                //cout << endl << "Frame skipped, watch out" << endl;

            string filename  = (it->second).c_str();
            int    size = filename.size();
            string timestamp =  filename.substr(0,size-4);
            ofs_traj  << timestamp << " "
                      << transl(0) << " " << transl(1) << " " << transl(2) << " "
                      << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w()
                      << std::endl;

            // access the pose of the camera via vo_->lastFrame()->T_f_w_.
            T_f_w = vo_->lastFrame()->T_f_w_.matrix();
            T_inc = T_f_w_prev * T_f_w.inverse();
            T_f_w_prev = T_f_w;
            scene.setPose( T_inc );
            scene.setText( vo_->lastFrame()->id_, vo_->lastProcessingTime()*1000, vo_->lastNumPtObservations(), 0, vo_->lastNumLsObservations(), 0 );
            cv::Mat dbg_img = vo_->get_debug_image();
            if( !dbg_img.empty() )
                scene.setImage( dbg_img );
            else
                scene.setImage("./empty_img.png");

            // introduce 3d features to the scene
            vector< Matrix<double,3,1> > points3d;
            for(list<PointFeat*>::iterator it=vo_->lastFrame()->pt_fts_.begin(); it!=vo_->lastFrame()->pt_fts_.end();++it)
                if((*it)->feat3D != NULL)
                   points3d.push_back( (*it)->feat3D->pos_ );
            scene.setPointsSVO(points3d);

            // introduce 3d line segments to the scene
            vector< Matrix<double,6,1> > lines3d;
            Matrix<double,6,1> line3d_;
            for(list<LineFeat*>::iterator it=vo_->lastFrame()->seg_fts_.begin(); it!=vo_->lastFrame()->seg_fts_.end(); ++it)
            {
              if((*it)->feat3D != NULL)
              {
                line3d_ << (*it)->feat3D->spos_, (*it)->feat3D->epos_;
                lines3d.push_back( line3d_ );
              }
            }
            scene.setLinesSVO(lines3d);

            // update scene
            scene.updateScene();

        }
        frame_counter++;

    }
    cout << endl << "End of the SVO pipeline" << endl << endl;

    std::cout << "Done." << std::endl;

    return 0;

}

int BenchmarkNode::runFromFolder(vk::ATANCamera* cam_, svo_options opts)
{
     std::cout<<"=========ATAN camera model==========="<<std::endl;

    // grab options
    int seq_offset     = opts.seq_offset;
    int seq_step       = opts.seq_step;
    int seq_length     = opts.seq_length;
    bool has_points    = opts.has_points;
    bool has_lines     = opts.has_lines;
    bool is_tum        = opts.is_tum;
    string dataset_dir = opts.dataset_dir;
    string images_dir  = opts.images_dir;
    string map_out     = opts.map_out;
    string traj_out    = opts.traj_out;
    int fps_           = 30;

    // Read content of the .yaml dataset configuration file
    YAML::Node dset_config = YAML::LoadFile(dataset_dir+"/dataset_params_dso.yaml");

    // get a sorted list of files in the img directory
    boost::filesystem::path img_dir_path(images_dir.c_str());
    if (!boost::filesystem::exists(img_dir_path))
    {
        cout << endl << "Image directory does not exist: \t" << images_dir << endl;
        return -1;
    }

    // get all files in the img directory
    size_t max_len = 0;
    std::list<std::string> imgs;
    boost::filesystem::directory_iterator end_itr; // default construction yields past-the-end
    for (boost::filesystem::directory_iterator file(img_dir_path); file != end_itr; ++file)
    {
        boost::filesystem::path filename_path = file->path().filename();
        if (boost::filesystem::is_regular_file(file->status()) &&
                (filename_path.extension() == ".png"  ||
                 filename_path.extension() == ".jpg"  ||
                 filename_path.extension() == ".jpeg" ||
                 filename_path.extension() == ".tiff") )
        {
            std::string filename(filename_path.string());
            imgs.push_back(filename);
            max_len = max(max_len, filename.length());
        }
    }

    // sort them by filename; add leading zeros to make filename-lengths equal if needed
    std::map<std::string, std::string> sorted_imgs;
    int n_imgs = 0;
    for (std::list<std::string>::iterator img = imgs.begin(); img != imgs.end(); ++img)
    {
        sorted_imgs[std::string(max_len - img->length(), '0') + (*img)] = *img;
        n_imgs++;
    }

    // add offset / step / length
    int seq_end = n_imgs;
    if( seq_length != 0 )
        int seq_end = std::min( seq_length*seq_step+seq_offset, n_imgs );
    std::map<std::string, std::string> sorted_imgs_aux = sorted_imgs;
    sorted_imgs.clear();
    int k = 0;
    for (auto img = sorted_imgs_aux.begin(); img != sorted_imgs_aux.end(); std::advance(img,seq_step) )
    {
        if( k >= seq_offset && k <= seq_end )
            sorted_imgs.insert( *img );
        k++;
    }

    // create scene
    sceneRepresentation scene("../app/scene_config.ini");
    Matrix<double,4,4> T_c_w, T_f_w = Matrix<double,4,4>::Identity(), T_f_w_prev = Matrix<double,4,4>::Identity(), T_inc;
    T_c_w = Matrix<double,4,4>::Identity();
    scene.initializeScene(T_f_w);

    // run SVO (pose estimation)
    std::list<FramePtr> frames;
    int frame_counter = 1;
    std::ofstream ofs_traj(traj_out.c_str());
    for (std::map<std::string, std::string>::iterator it = sorted_imgs.begin(); it != sorted_imgs.end(); ++it )
    {
        // load image
        boost::filesystem::path img_path = img_dir_path / boost::filesystem::path(it->second.c_str());
        if (frame_counter == 1)
            std::cout << "reading image " << img_path.string() << std::endl;
        cv::Mat img(cv::imread(img_path.string(), CV_8UC1));
        // IMPORTANT: The image must be flipped if focal length is negative
        // since the optimization code assumes that both f_x and f_y are positive
        {
          // get camera config
          YAML::Node cam_config = dset_config["cam0"];
          // check f_y sign
          if(cam_config["cam_fy"].as<double>()<0)
            cv::flip(img, img, 0); // Vertical flipping (around x axis, changes y coordinate)
        }
        assert(!img.empty());

        // undistort image
        cv::Mat img_rec;
        std::cout<<"test1============="<<std::endl;
        img_rec = undistortFishEye(img,0.93);
        cv::imwrite("atan_img.jpg",img_rec);

        // process frame
        // vo_->addImage(img, frame_counter / (double)fps_);
        vo_->addImage(img_rec, frame_counter / (double)fps_);

        // display tracking quality
        if (vo_->lastFrame() != NULL) {
            std::cout << "Frame-Id: "       << vo_->lastFrame()->id_ << " \t"
                      << "#PointFeatures: " << vo_->lastNumPtObservations() << " \t"
                      << "#LineFeatures: "  << vo_->lastNumLsObservations() << " \t"
                      << "Proc. Time: "     << vo_->lastProcessingTime()*1000 << "ms" << std::endl << std::endl;
            frames.push_back(vo_->lastFrame());

            // save trajectory
            Eigen::Matrix<double, 6, 6> cov = vo_->lastFrame()->Cov_;
            bool skip_frame = false;
            for (int i = 0; i < 6; i++)
                for (int j = 0; j < 6; j++)
                    if (! ((1.e-16 < fabs(cov(i,j))) && (fabs(cov(i,j)) < 1.e+16)) )    // likely an invalid pose -> seriously?
                        skip_frame = true;

            // access the pose of the camera via vo_->lastFrame()->T_f_w_.
            Sophus::SE3 world_transf = vo_->lastFrame()->T_f_w_.inverse();
            Eigen::Quaterniond quat  = world_transf.unit_quaternion();
            Eigen::Vector3d transl   = world_transf.translation();
            Matrix3d rot = world_transf.rotation_matrix();
            if ( ((transl(0) == 0.) && (transl(1) == 0.) && (transl(2) == 0.)) &&
                    ((quat.x() == -0.) && (quat.y() == -0.) && (quat.z() == -0.) && (quat.w() == 1.)) )
                skip_frame = true;

            if (skip_frame)
                continue;
                //cout << endl << "Frame skipped, watch out" << endl;

            string filename  = (it->second).c_str();
            int    size = filename.size();
            string timestamp =  filename.substr(0,size-4);
            ofs_traj  << timestamp << " "
                      << transl(0) << " " << transl(1) << " " << transl(2) << " "
                      << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.w()
                      << std::endl;

            // access the pose of the camera via vo_->lastFrame()->T_f_w_.
            T_f_w = vo_->lastFrame()->T_f_w_.matrix();
            T_inc = T_f_w_prev * T_f_w.inverse();
            T_f_w_prev = T_f_w;
            scene.setPose( T_inc );
            scene.setText( vo_->lastFrame()->id_, vo_->lastProcessingTime()*1000, vo_->lastNumPtObservations(), 0, vo_->lastNumLsObservations(), 0 );
            cv::Mat dbg_img = vo_->get_debug_image();
            if( !dbg_img.empty() )
                scene.setImage( dbg_img );
            else
                scene.setImage("./empty_img.png");

            // introduce 3d features to the scene
            vector< Matrix<double,3,1> > points3d;
            for(list<PointFeat*>::iterator it=vo_->lastFrame()->pt_fts_.begin(); it!=vo_->lastFrame()->pt_fts_.end();++it)
                if((*it)->feat3D != NULL)
                   points3d.push_back( (*it)->feat3D->pos_ );
            scene.setPointsSVO(points3d);

            // introduce 3d line segments to the scene
            vector< Matrix<double,6,1> > lines3d;
            Matrix<double,6,1> line3d_;
            for(list<LineFeat*>::iterator it=vo_->lastFrame()->seg_fts_.begin(); it!=vo_->lastFrame()->seg_fts_.end(); ++it)
            {
              if((*it)->feat3D != NULL)
              {
                line3d_ << (*it)->feat3D->spos_, (*it)->feat3D->epos_;
                lines3d.push_back( line3d_ );
              }
            }
            scene.setLinesSVO(lines3d);

            // update scene
            scene.updateScene();

        }
        frame_counter++;

    }
    cout << endl << "End of the SVO pipeline" << endl << endl;

    std::cout << "Done." << std::endl;

    return 0;

}

} // namespace plsvo

const cv::String keys =
    "{help h usage ? |                  | print this message   }"
    "{@dset          |sin2_tex2_h1_v8_d | dataset folder inside $SVO_DATASET_DIR }"
    "{expname        |<none>            | name of the experiment }"
    "{seqoff seqo    |0                 | start position in the sequence }"
    "{seql seqlength |0                | number of frames to test (from 1st ref frame) }"
    "{seqs seqstep   |1                 | step size in sequence (nr of frames to advance) }"
    "{verbose v      |                  | show more verbose information in optimization (akin debug) }"
    "{display-optim disp-optim|false    | display sparse residue images in optimization }"
    "{display disp          |false      | display residue images (debug) }"
    "{init initialization   |2          | value to take as initialization of transformation }"
    "{haspt haspoints       |true       | bool to employ or not point }"
    "{hasls haslines        |true      | bool to employ or not line segments }"
    "{mapout                |<none>     | name of the pcd output file for the map }"
    "{trajout               |trajout.txt | name of the output file for the trajectory }"
    "{files               |../config/dataset_path_ceiling.txt | name of the input file for the hole pipline }"
    "{calib_path          |../config | path of the calib file }"
    "{calib               |/dataset_params_ceiling.yaml | name of the calib file }"
    ;

// Examples of use:
// sin2_tex2_h1_v8_d            ./run_pipeline_comp sin2_tex2_h1_v8_d
// ICL-NUIM                     ./run_pipeline_comp ICL-NUIM/lrkt0-retinex
// ASL_Dataset                  ./run_pipeline_comp ASL_Dataset/MH_01_easy

void parseArgument(char* arg)// 分析参数
{
	int option;
	float foption;
	char buf[1000];

    if(1==sscanf(arg,"files=%d",&option))
    {
        if(option==1)
        {
            //files = true;
            files = option;
            printf("=====get files ,all image and calib files====== \n");
        }
        return;
    }
	// printf("could not parse argument \"%s\"!!!!\n", arg);
}

cv::Mat undistortFishEye(const cv::Mat &distorted, const float w)
{
    cv::Mat map_x, map_y;
    map_x.create(distorted.size(), CV_32FC1);
    map_y.create(distorted.size(), CV_32FC1);

    double Cx = distorted.cols / 2.0;
    double Cy = distorted.rows / 2.0;

    for (double x = -1.0; x < 1.0; x += 1.0/Cx) {
        for (double y = -1.0; y < 1.0; y += 1.0/Cy) {
            double ru = sqrt(x*x + y*y);
            double rd = (1.0 / w)*atan(2.0*ru*tan(w / 2.0));

            map_x.at<float>(y*Cy + Cy, x*Cx + Cx) = rd/ru * x*Cx + Cx;
            map_y.at<float>(y*Cy + Cy, x*Cx + Cx) = rd/ru * y*Cy + Cy;
        }
    }

    cv::Mat undistorted;
    remap(distorted, undistorted, map_x, map_y, CV_INTER_LINEAR);
    return undistorted;
}

int main(int argc, char** argv)
{
    std::cout<<"==========test_start=========="<<std::endl;
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("SVO test: run_pipeline");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    std::string dataset_name = parser.get<std::string>(0);
    std::string experiment_name, map_output, traj_output;
    if(parser.has("expname"))
      experiment_name = parser.get<std::string>("expname");
    else
      experiment_name  = dataset_name;
    if(parser.has("mapout"))
      map_output = parser.get<std::string>("mapout");
    else
      map_output  = "map_out.pcd";
    traj_output = parser.get<std::string>("trajout");

    files = parser.get<std::string>("files");
    calib = parser.get<std::string>("calib");
    calib_path = parser.get<std::string>("calib_path");

    int init           = parser.get<int>("init");
    bool verbose       = parser.has("verbose");
    bool display_optim = parser.has("disp-optim");
    bool display       = parser.has("disp");

    svo_options opts;
    opts.seq_offset = parser.get<int>("seqoff");
    opts.seq_step   = parser.get<int>("seqs");
    opts.seq_length = parser.get<int>("seql");
    opts.has_points = parser.get<bool>("haspt");
    opts.has_lines  = parser.get<bool>("hasls");
    opts.traj_out   = traj_output;
    opts.map_out    = map_output;

    opts.is_tum = false;
    if( dataset_name.find("TUM") != std::string::npos )
    {
        opts.is_tum = true;
    }

    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    std::string dataset_dir = calib_path;
    string img_dir;
   

    ifstream imgfile;
    imgfile.open(files,ios::in);    
    getline(imgfile,img_dir,'\n');
    imgfile.close();

    std::cout<<"==========“calib_dir”=========="<<dataset_dir+calib<<std::endl;
    YAML::Node dset_config = YAML::LoadFile(dataset_dir+calib);
    opts.dataset_dir = dataset_dir;
    
    std::cout<<"==========“img_dir”=========="<<img_dir<<std::endl;
    opts.images_dir = img_dir;

    // Setup camera and run node
    YAML::Node cam_config = dset_config["cam0"];
    string camera_model = cam_config["cam_model"].as<string>();
    // std::cout<<"===================="<<dset_config<<std::endl;
    if( camera_model == "Pinhole" )//针孔相机模型
    {   
        // setup cameras
        vk::PinholeCamera* cam_pin;
        vk::PinholeCamera* cam_pin_und;
        cam_pin = new vk::PinholeCamera(//有畸变模型
            cam_config["cam_width"].as<double>(),
            cam_config["cam_height"].as<double>(),
            fabs(cam_config["cam_fx"].as<double>()),
            fabs(cam_config["cam_fy"].as<double>()),
            cam_config["cam_cx"].as<double>(),
            cam_config["cam_cy"].as<double>(),
            cam_config["cam_d0"].as<double>(),
            cam_config["cam_d1"].as<double>(),
            cam_config["cam_d2"].as<double>(),
            cam_config["cam_d3"].as<double>()  );
        cam_pin_und = new vk::PinholeCamera(//无畸变模型
            cam_config["cam_width"].as<double>(),
            cam_config["cam_height"].as<double>(),
            fabs(cam_config["cam_fx"].as<double>()),
            fabs(cam_config["cam_fy"].as<double>()),
            cam_config["cam_cx"].as<double>(),
            cam_config["cam_cy"].as<double>() );
        // Set options for FrameHandlerMono
        plsvo::FrameHandlerMono::Options handler_opts(opts.has_points,opts.has_lines);//处理器，处理点线// 选择调动点和线
        plsvo::BenchmarkNode benchmark(cam_pin_und);//基准模型
        // run pipeline
        benchmark.runFromFolder(cam_pin,opts);
    }
    else if( camera_model == "ATAN" || camera_model == "Atan" )//？？？什么相机模型
    {
        // setup cameras
        vk::ATANCamera* cam_atan;
        vk::ATANCamera* cam_atan_und;
        cam_atan = new vk::ATANCamera(
            cam_config["cam_width"].as<double>(),
            cam_config["cam_height"].as<double>(),
            fabs(cam_config["cam_fx"].as<double>()),
            fabs(cam_config["cam_fy"].as<double>()),
            cam_config["cam_cx"].as<double>(),
            cam_config["cam_cy"].as<double>(),
            cam_config["cam_d0"].as<double>()  );
        cam_atan_und = new vk::ATANCamera(
            cam_config["cam_width"].as<double>(),
            cam_config["cam_height"].as<double>(),
            fabs(cam_config["cam_fx"].as<double>()),
            fabs(cam_config["cam_fy"].as<double>()),
            cam_config["cam_cx"].as<double>(),
            cam_config["cam_cy"].as<double>(),
            cam_config["cam_d0"].as<double>()  );
        // Set options for FrameHandlerMono
        plsvo::FrameHandlerMono::Options handler_opts(opts.has_points,opts.has_lines);
        plsvo::BenchmarkNode benchmark(cam_atan_und);
        // run pipeline
        benchmark.runFromFolder(cam_atan,opts);
    }

    printf("BenchmarkNode finished.\n");
    return 0;

}
