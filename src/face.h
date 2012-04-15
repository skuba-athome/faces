#include <cv.h>
#include <cxcore.h>
#include <cvaux.h>
#include <highgui.h>
#include <string.h>
#include <stdlib.h>


void convertmsg2img(const sensor_msgs::ImageConstPtr& msg);
IplImage * detect_face(char filename[]);
//// Function prototypes
void learn();
void recognize();
void doPCA();
void storeTrainingData();
int  loadTrainingData(CvMat ** pTrainPersonNumMat);
int  findNearestNeighbor(float * projectedTestFace);
int  loadFaceImgArray(char * filename);
void printUsage();
IplImage* cropImage(const IplImage *img, const CvRect region);
IplImage* resizeImage(const IplImage *origImg, int newWidth, int newHeight);
void recognize_realtime();
