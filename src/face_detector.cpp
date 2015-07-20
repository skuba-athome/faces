#include <stdlib.h>
#include <ros/ros.h>
#include <ros/package.h>

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/String.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

class FaceDetector
{
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;
    ros::Publisher face_pub_;

    cv::CascadeClassifier face_cascade;
    cv::Mat gray_im;
    std::vector<cv::Rect> faces;

    int img_id;
    char comm[255];

    public:
        
    FaceDetector() : it_(nh_)
    {
        std::string face_cascade_filename;
        const std::string face_cascade_default_filename = ros::package::getPath("faces") + "/model/haarcascade_frontalface_default.xml";
        nh_.param("face_cascade_file", face_cascade_filename, face_cascade_default_filename);

        face_cascade.load(face_cascade_filename);
        img_id = 0;
        system("mkdir /run/shm/face");
    
        image_sub_ = it_.subscribe("image", 1, &FaceDetector::process, this);
        image_pub_ = it_.advertise("output", 1);
    }

    ~FaceDetector()
    {
    }

    void process(const sensor_msgs::ImageConstPtr& msg)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        cv::cvtColor(cv_ptr->image, gray_im, CV_BGR2GRAY);
        //cv::resize(gray_im, gray_im, cv::Size(), 0.5, 0.5);
        //cv::equalizeHist(gray_im, gray_im);

        //if(face_cascade.isOldFormatCascade()) printf("Old Cascade\n");
        //else printf("new Cascade\n");

        face_cascade.detectMultiScale( gray_im, faces, 1.5, 3, 0|CV_HAAR_DO_CANNY_PRUNING, cv::Size(50, 50) );

        for( size_t i = 0; i < faces.size(); ++i )
        {
            ++img_id;
            cv::Mat crop = cv_ptr->image(faces[i]);
            sprintf(comm, "/run/shm/face/%d.jpg", img_id);
            cv::imwrite(comm, crop);
            cv::rectangle(cv_ptr->image, faces[i], cv::Scalar(0, 0, 255), 5);
        }

        // Output modified video stream
        image_pub_.publish(cv_ptr->toImageMsg());
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "face_detector");
    FaceDetector fd;
    ros::spin();
    return 0;
}

