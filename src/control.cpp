#include "ros/ros.h"
#include "std_msgs/String.h"

#include <sstream>
#include <cstdio>

int main(int argc, char **argv)
{
  char s[100];
  ros::init(argc, argv, "talker");
  ros::NodeHandle n;
  ros::Publisher control_pub = n.advertise<std_msgs::String>("control", 10);

  ros::Rate loop_rate(10);
  while (ros::ok())
  {
    scanf("%s",s);

    std_msgs::String msg;

    std::stringstream ss;

    ss << s;

    msg.data = ss.str();

    ROS_INFO("%s", msg.data.c_str());

    control_pub.publish(msg);

    ros::spinOnce();

    loop_rate.sleep();

  }


  return 0;
}
