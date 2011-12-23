#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/Image.h>
#include <cv.h>
#include <highgui.h>
#include <opencv2/objdetect/objdetect.hpp>
#include <std_msgs/String.h>
#include <string.h>
#include <cxcore.h>
#include <cvaux.h>
#include <stdlib.h>

//#define RGB
using namespace std;
using namespace cv;
using namespace cv_bridge;

#define MAX_FACES 20
#define TOPIC_CONTROL "/cmd_state"
#ifdef RGB
IplImage* img = cvCreateImage( cvSize(640,480),IPL_DEPTH_8U, 3 );
#else
IplImage* img = cvCreateImage( cvSize(640,480),IPL_DEPTH_8U, 1 );
#endif
cv::Mat depthImg ;
cv_bridge::CvImagePtr bridge;

//// Global variables
IplImage ** faceImgArr        = 0; // array of face images
CvMat    *  personNumTruthMat = 0; // array of person numbers
int nTrainFaces               = 0; // the number of training images
int nEigens                   = 0; // the number of eigenvalues
IplImage * pAvgTrainImg       = 0; // the average image
IplImage ** eigenVectArr      = 0; // eigenvectors
CvMat * eigenValMat           = 0; // eigenvalues
CvMat * projectedTrainFaceMat = 0; // projected training faces
IplImage * faceImg;
int faceCount = 0;
int chkSave = 0;
int nNames = 0;
char name[100];
double min_range_;
double max_range_;
float dist[480][640];
int canPrintDepth = 0;

int g_nearest[20];
int g_count = 0;
int is_recog = 0;
int is_init = 0;

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


void kinectCallBack(const sensor_msgs::ImageConstPtr& msg)
{
  //ROS_INFO("[%d] [%d]",msg->width,msg->height);
  	convertmsg2img(msg);
  	cvEqualizeHist(img,img);
  	detect_face("haarcascade_frontalface_alt.xml");
  // show image
  	if(is_recog){
		g_count++;
	    recognize_realtime();
	}
  	cvShowImage("test",img);

  	cv::waitKey(10);
}

void depthCb( const sensor_msgs::ImageConstPtr& image )
{
  canPrintDepth = 0;
    // convert to cv image
    try
    {
        bridge = cv_bridge::toCvCopy(image, "32FC1");
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Failed to transform depth image.");
        return;
    }

    // convert to something visible
    depthImg = Mat(bridge->image.rows, bridge->image.cols, CV_8UC1);
  //  printf("[%d] [%d] \n",bridge->image.rows, bridge->image.cols);
    for(int i = 0; i < bridge->image.rows; i++)
    {
        float* Di = bridge->image.ptr<float>(i);
        char* Ii = depthImg.ptr<char>(i);
        for(int j = 0; j < bridge->image.cols; j++)
        {
            Ii[j] = (char) (255*((Di[j]-min_range_)/(max_range_-min_range_)));
            dist[i][j] = Di[j];
        }
    }
    // display
    canPrintDepth = 1;
   // cv::imshow("dept_test", depthImg);
  //  cv::waitKey(3);
}

void controlCallBack(const std_msgs::String::ConstPtr& msg)
{
  ROS_INFO("%s",msg->data.c_str());

	// debug recognizer 
	//return ;

  if(!strcmp(msg->data.c_str(),"test"))
  {
 	is_recog = 1; 
    return ;
  }
  if(!strcmp(msg->data.c_str(),"target"))
    {
      learn();
      return ;
    }
  /*if(strcmp(msg->data.c_str(),"one")!=0 && strcmp(msg->data.c_str(),"peter")!=0)
     return ;*/
  chkSave = 1;
  FILE * imgListFile;
  // open the input file
  if( !(imgListFile = fopen("./data/names.txt", "a+")) )
  {
          //fprintf(stderr, "Can\'t open file %s\n", filename);
          printf("ERROR : CAN'T OPEN train.txt\n");
          //return 0;
  }

  strcpy(name,msg->data.c_str());
  //strcpy(name,"pond");
  fprintf(imgListFile,"%d %s\n",nNames++,name);

  fclose(imgListFile);
}
int main(int argc,char * argv[])
{
  FILE * imgListFile = 0;
  ros::init(argc,argv,"faces");
  ros::NodeHandle n;
  ros::NodeHandle nh("~");

  nh.param("min_range", min_range_, 0.5);
  nh.param("max_range", max_range_, 5.5);

#ifdef RGB
  ros::Subscriber sub = n.subscribe("/camera/rgb/image_color", 1, kinectCallBack);
#else
  ros::Subscriber sub = n.subscribe("/camera/rgb/image_mono", 1, kinectCallBack);
#endif
  ros::Subscriber sub2 = n.subscribe(TOPIC_CONTROL, 1, controlCallBack);

  ros::Subscriber subDepth = n.subscribe("/camera/depth/image",1,depthCb);


  if( !(imgListFile = fopen("data/names.txt", "a+")) )
  {
          fprintf(stderr, "Can\'t open file %s\n", "names.txt");
          return 0;
  }

  // count the number of faces
  char tmp[20];
  while( fgets(tmp, 512, imgListFile) ) ++nNames;

  printf("pass\n");
  learn();
  system("espeak --stdout \'Ready\' | aplay");
  is_init = 1;
  ros::spin();
}

void convertmsg2img(const sensor_msgs::ImageConstPtr& msg)
{
  for(int i=0;i<640*480;i++)
    {
#ifdef RGB
            img->imageData[i*3] = msg->data[i*3+2];
            img->imageData[i*3+1] = msg->data[i*3+1];
            img->imageData[i*3+2] = msg->data[i*3];
#else
		if( dist[i/640][i%640] < 2.0f || 1){
            img->imageData[i] = msg->data[i];
		}else
		{
			img->imageData[i] = 0;
		}
#endif
    }
}

IplImage * detect_face(char filename[]){

  CvMemStorage *storage = cvCreateMemStorage( 0 );
  CvHaarClassifierCascade *cascade  = ( CvHaarClassifierCascade* )cvLoad( filename ,0 , 0, 0 );
  CvRect *r = 0;

  if(cascade == NULL)
  {
    printf("can't open haarcascade file . \n");
    return 0;
  }

  double t = (double)cvGetTickCount();

  CvSeq* faces = cvHaarDetectObjects( img , cascade , storage , 1.1 , 2 , CV_HAAR_DO_CANNY_PRUNING , cvSize(60,60));

  float f_min = 4.0f;
  for ( int i=0;i<( faces ? faces->total:0);i++)
  {;
          CvRect* tmp = (CvRect*)cvGetSeqElem(faces,i);
          if( dist[tmp->y+tmp->height/2][tmp->x+tmp->width/2] < f_min )
          {
            r = tmp;
			f_min = dist[tmp->y+tmp->height/2][tmp->x+tmp->width/2]; 
          }
  }

  //printf("%.2f \n",bridge->image.ptr<float>( (r->y+r->height)*640 + r->x + r->width ));
  if(r == NULL ) // check for can't find
  {
    printf("can't find any faces in Image\n");
    return 0;
  }
  //if(canPrintDepth)   printf("distance face : %.2f \n",dist[r->y+r->height/2][r->x+r->width/2]);
  //draw_image
  cvRectangle(img,cvPoint(r->x,r->y),cvPoint(r->x+r->width,r->y+r->height),cvScalarAll(0.5),5,2,0);
  //cvRectangle(depthImg,cvPoint(r->x,r->y),cvPoint(r->x+r->width,r->y+r->height),cvScalarAll(0.5),5,2,0);
  //cv::rectangle(depthImg,cvPoint(r->x,r->y),cvPoint(r->x+r->width,r->y+r->height),cvScalarAll(1.0),255,1,0);
  faceImg = cropImage(img, *r);
  faceImg = resizeImage(faceImg,100,100);
  // histogram equalized
  cvEqualizeHist(faceImg,faceImg);

  if(chkSave)
  {
    FILE * imgListFile;
    // open the input file
    if( !(imgListFile = fopen("./data/train.txt", "a+")) )
    {
            //fprintf(stderr, "Can\'t open file %s\n", filename);
            printf("ERROR : CAN'T OPEN train.txt\n");
            //return 0;
    }
    fprintf(imgListFile,"%d data/%s_%d.pgm\n",nNames-1,name,faceCount);
    fclose(imgListFile);

    char cstr[100];
    sprintf(cstr, "./data/%s_%d.pgm",name , faceCount++);
    printf("Storing the current face of '%s' into image '%s'.\n", name , cstr);
    cvSaveImage(cstr, faceImg, NULL);
    if(faceCount == MAX_FACES)
    {
      chkSave = 0;
      faceCount = 0;
	learn();
    }


  }

 // cvShowImage("face",faceImg);
  t = (double)cvGetTickCount() - t ;
  // clear memory
  if(storage) cvReleaseMemStorage(&storage);
  if(cascade) cvReleaseHaarClassifierCascade(&cascade);

  //printf("detect use time : %g\n",t/1000);

  return faceImg;
}


IplImage* cropImage(const IplImage *img, const CvRect region)
{
        IplImage *imageTmp;
        IplImage *imageRGB;
        CvSize size;
        size.height = img->height;
        size.width = img->width;

        if (img->depth != IPL_DEPTH_8U) {
                printf("ERROR in cropImage: Unknown image depth of %d given in cropImage() instead of 8 bits per pixel.\n", img->depth);
                exit(1);
        }

        // First create a new (color or greyscale) IPL Image and copy contents of img into it.
        imageTmp = cvCreateImage(size, IPL_DEPTH_8U, img->nChannels);
        cvCopy(img, imageTmp, NULL);

        // Create a new image of the detected region
        // Set region of interest to that surrounding the face
        cvSetImageROI(imageTmp, region);
        // Copy region of interest (i.e. face) into a new iplImage (imageRGB) and return it
        size.width = region.width;
        size.height = region.height;
        imageRGB = cvCreateImage(size, IPL_DEPTH_8U, img->nChannels);
        cvCopy(imageTmp, imageRGB, NULL);       // Copy just the region.

    cvReleaseImage( &imageTmp );
        return imageRGB;
}

IplImage* resizeImage(const IplImage *origImg, int newWidth, int newHeight)
{
        IplImage *outImg = 0;
        int origWidth;
        int origHeight;
        if (origImg) {
                origWidth = origImg->width;
                origHeight = origImg->height;
        }
        if (newWidth <= 0 || newHeight <= 0 || origImg == 0 || origWidth <= 0 || origHeight <= 0) {
                printf("ERROR in resizeImage: Bad desired image size of %dx%d\n.", newWidth, newHeight);
                exit(1);
        }

        // Scale the image to the new dimensions, even if the aspect ratio will be changed.
        outImg = cvCreateImage(cvSize(newWidth, newHeight), origImg->depth, origImg->nChannels);
        if (newWidth > origImg->width && newHeight > origImg->height) {
                // Make the image larger
                cvResetImageROI((IplImage*)origImg);
                cvResize(origImg, outImg, CV_INTER_LINEAR);     // CV_INTER_CUBIC or CV_INTER_LINEAR is good for enlarging
        }
        else {
                // Make the image smaller
                cvResetImageROI((IplImage*)origImg);
                cvResize(origImg, outImg, CV_INTER_AREA);       // CV_INTER_AREA is good for shrinking / decimation, but bad at enlarging.
        }

        return outImg;
}
//////////////////////////////////
// learn()
//
void learn()
{
        int i, offset;

        // load training data
        nTrainFaces = loadFaceImgArray("data/train.txt");
        printf("%d\n",nTrainFaces);
        if( nTrainFaces < 2 )
        {
                fprintf(stderr,
                        "Need 2 or more training faces\n"
                        "Input file contains only %d\n", nTrainFaces);
                return;
        }
        printf("debug : learn() , load image");
        // do PCA on the training faces
        doPCA();

        // project the training images onto the PCA subspace
        projectedTrainFaceMat = cvCreateMat( nTrainFaces, nEigens, CV_32FC1 );
        offset = projectedTrainFaceMat->step / sizeof(float);
        for(i=0; i<nTrainFaces; i++)
        {
                //int offset = i * nEigens;
                cvEigenDecomposite(
                        faceImgArr[i],
                        nEigens,
                        eigenVectArr,
                        0, 0,
                        pAvgTrainImg,
                        //projectedTrainFaceMat->data.fl + i*nEigens);
                        projectedTrainFaceMat->data.fl + i*offset);
        }

        // store the recognition data as an xml file
    storeTrainingData();
	if(is_init)
    	system("espeak --stdout \'now i remember you\' | aplay");
}


//////////////////////////////////
// recognize()
//
void recognize()
{
        int i, nTestFaces  = 0;         // the number of test images
        CvMat * trainPersonNumMat = 0;  // the person numbers during training
        float * projectedTestFace = 0;

        // load test images and ground truth for person number
        nTestFaces = loadFaceImgArray("test.txt");
        printf("%d test faces loaded\n", nTestFaces);

        // load the saved training data
        if( !loadTrainingData( &trainPersonNumMat ) ) return;

        // project the test images onto the PCA subspace
        projectedTestFace = (float *)cvAlloc( nEigens*sizeof(float) );
        for(i=0; i<nTestFaces; i++)
        {
                int iNearest, nearest, truth;

                // project the test image onto the PCA subspace
                cvEigenDecomposite(
                        faceImgArr[i],
                        nEigens,
                        eigenVectArr,
                        0, 0,
                        pAvgTrainImg,
                        projectedTestFace);

                iNearest = findNearestNeighbor(projectedTestFace);
                truth    = personNumTruthMat->data.i[i];
                nearest  = trainPersonNumMat->data.i[iNearest];

                printf("nearest = %d, Truth = %d\n", nearest, truth);
		//   "espeak --stdout \'" + sp + "' | aplay"
			char name[100];
			int num_tmp;
			FILE *fp = fopen("data/names.txt", "r");
			for(int fi=0;fi<nearest;fi++)			
				fscanf(fp, "%d %s", &num_tmp , name);	// read number of objects
			fclose(fp);
			printf("your name is name : %s\n",name);
			char cmd[1024];
			sprintf(cmd,  "espeak --stdout \' %s \' | aplay", name);
			system(cmd);

        }
}

void recognize_realtime()
{
  int i, nTestFaces  = 0;         // the number of test images
  CvMat * trainPersonNumMat = 0;  // the person numbers during training
  float * projectedTestFace = 0;

  // load the saved training data
  if( !loadTrainingData( &trainPersonNumMat ) ) return;
  if(!faceImg)
  {
    printf("debug : can't find face for recog\n");
    return ;
  }
  // project the test images onto the PCA subspace
  projectedTestFace = (float *)cvAlloc( nEigens*sizeof(float) );

          int iNearest, nearest, truth;

          // project the test image onto the PCA subspace
          cvEigenDecomposite(
                  faceImg,
                  nEigens,
                  eigenVectArr,
                  0, 0,
                  pAvgTrainImg,
                  projectedTestFace);

          iNearest = findNearestNeighbor(projectedTestFace);
          truth    = personNumTruthMat->data.i[i];
          nearest  = trainPersonNumMat->data.i[iNearest];

	printf("nearest = %d, Truth = %d\n", nearest, truth);
	g_nearest[nearest]++;
	if(g_count == 10){
		int index_max=0;
		int max = -1;
		for(int i = 0 ; i< 10 ; i++)
		{
			if(g_nearest[i] > max)
			{
				max = g_nearest[i];
				index_max = i;
			}
		}
		char name[100];
		int num_tmp;
		FILE *fp = fopen("data/names.txt", "r");
		for(int fi=0;fi<=index_max;fi++)			
			fscanf(fp, "%d %s", &num_tmp , name);	// read number of objects
		fclose(fp);
		printf("your name is name : %s\n",name);
		char cmd[1024];
		sprintf(cmd,  "espeak --stdout -s 150 \' your name is %s \' | aplay", name);
		system(cmd);
		is_recog = 0;
		g_count = 0;
		for(int i =0;i<10;i++)	g_nearest[i]=0;
	}
}


//////////////////////////////////
// loadTrainingData()
//
int loadTrainingData(CvMat ** pTrainPersonNumMat)
{
        CvFileStorage * fileStorage;
        int i;

        // create a file-storage interface
        fileStorage = cvOpenFileStorage( "facedata.xml", 0, CV_STORAGE_READ );
        if( !fileStorage )
        {
                fprintf(stderr, "Can't open facedata.xml\n");
                return 0;
        }

        nEigens = cvReadIntByName(fileStorage, 0, "nEigens", 0);
        nTrainFaces = cvReadIntByName(fileStorage, 0, "nTrainFaces", 0);
        *pTrainPersonNumMat = (CvMat *)cvReadByName(fileStorage, 0, "trainPersonNumMat", 0);
        eigenValMat  = (CvMat *)cvReadByName(fileStorage, 0, "eigenValMat", 0);
        projectedTrainFaceMat = (CvMat *)cvReadByName(fileStorage, 0, "projectedTrainFaceMat", 0);
        pAvgTrainImg = (IplImage *)cvReadByName(fileStorage, 0, "avgTrainImg", 0);
        eigenVectArr = (IplImage **)cvAlloc(nTrainFaces*sizeof(IplImage *));
        for(i=0; i<nEigens; i++)
        {
                char varname[200];
                sprintf( varname, "eigenVect_%d", i );
                eigenVectArr[i] = (IplImage *)cvReadByName(fileStorage, 0, varname, 0);
        }

        // release the file-storage interface
        cvReleaseFileStorage( &fileStorage );

        return 1;
}
//////////////////////////////////
// storeTrainingData()
//
void storeTrainingData()
{
        CvFileStorage * fileStorage;
        int i;

        // create a file-storage interface
        fileStorage = cvOpenFileStorage( "facedata.xml", 0, CV_STORAGE_WRITE );

        // store all the data
        cvWriteInt( fileStorage, "nEigens", nEigens );
        cvWriteInt( fileStorage, "nTrainFaces", nTrainFaces );
        cvWrite(fileStorage, "trainPersonNumMat", personNumTruthMat, cvAttrList(0,0));
        cvWrite(fileStorage, "eigenValMat", eigenValMat, cvAttrList(0,0));
        cvWrite(fileStorage, "projectedTrainFaceMat", projectedTrainFaceMat, cvAttrList(0,0));
        cvWrite(fileStorage, "avgTrainImg", pAvgTrainImg, cvAttrList(0,0));
        for(i=0; i<nEigens; i++)
        {
                char varname[200];
                sprintf( varname, "eigenVect_%d", i );
                cvWrite(fileStorage, varname, eigenVectArr[i], cvAttrList(0,0));
        }

        // release the file-storage interface
        cvReleaseFileStorage( &fileStorage );
}


//////////////////////////////////
// findNearestNeighbor()
//
int findNearestNeighbor(float * projectedTestFace)
{
        //double leastDistSq = 1e12;
        double leastDistSq = DBL_MAX;
        int i, iTrain, iNearest = 0;

        for(iTrain=0; iTrain<nTrainFaces; iTrain++)
        {
                double distSq=0;

                for(i=0; i<nEigens; i++)
                {
                        float d_i =
                                projectedTestFace[i] -
                                projectedTrainFaceMat->data.fl[iTrain*nEigens + i];
                        //distSq += d_i*d_i / eigenValMat->data.fl[i];  // Mahalanobis
                        distSq += d_i*d_i; // Euclidean
                }

                if(distSq < leastDistSq)
                {
                        leastDistSq = distSq;
                        iNearest = iTrain;
                }
        }

        return iNearest;
}


//////////////////////////////////
// doPCA()
//
void doPCA()
{
        int i;
        CvTermCriteria calcLimit;
        CvSize faceImgSize;

        // set the number of eigenvalues to use
        nEigens = nTrainFaces-1;

        // allocate the eigenvector images
        faceImgSize.width  = faceImgArr[0]->width;
        faceImgSize.height = faceImgArr[0]->height;
        eigenVectArr = (IplImage**)cvAlloc(sizeof(IplImage*) * nEigens);
        for(i=0; i<nEigens; i++)
                eigenVectArr[i] = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

        // allocate the eigenvalue array
        eigenValMat = cvCreateMat( 1, nEigens, CV_32FC1 );

        // allocate the averaged image
        pAvgTrainImg = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

        // set the PCA termination criterion
        calcLimit = cvTermCriteria( CV_TERMCRIT_ITER, nEigens, 1);

        // compute average image, eigenvalues, and eigenvectors
        cvCalcEigenObjects(
                nTrainFaces,
                (void*)faceImgArr,
                (void*)eigenVectArr,
                CV_EIGOBJ_NO_CALLBACK,
                0,
                0,
                &calcLimit,
                pAvgTrainImg,
                eigenValMat->data.fl);

        cvNormalize(eigenValMat, eigenValMat, 1, 0, CV_L1, 0);
}


//////////////////////////////////
// loadFaceImgArray()
//
int loadFaceImgArray(char * filename)
{
        FILE * imgListFile = 0;
        char imgFilename[512];
        int iFace, nFaces=0;


        // open the input file
      //  printf("%s\n",filename);
        if( !(imgListFile = fopen(filename, "r")) )
        {
                fprintf(stderr, "Can\'t open file %s\n", filename);
                return 0;
        }

        // count the number of faces
        while( fgets(imgFilename, 512, imgListFile) ) ++nFaces;
        rewind(imgListFile);
        //printf("%d\n",nFaces);
        // allocate the face-image array and person number matrix
        faceImgArr        = (IplImage **)cvAlloc( nFaces*sizeof(IplImage *) );
        personNumTruthMat = cvCreateMat( 1, nFaces, CV_32SC1 );
        // store the face images in an array
        for(iFace=0; iFace<nFaces; iFace++)
        {
              //  printf("%d ",iFace);
                // read person number and name of image file
                fscanf(imgListFile,
                       "%d %s", personNumTruthMat->data.i+iFace, imgFilename);
                //sprintf(imgFilename,"%s_%d.pgm","pond",iFace);
                // load the face image
             //   printf("%s\n",imgFilename);
                faceImgArr[iFace] = cvLoadImage(imgFilename, CV_LOAD_IMAGE_GRAYSCALE);

                if( !faceImgArr[iFace] )
                {
                        fprintf(stderr, "Can\'t load image from %s\n", imgFilename);
                        return 0;
                }
        }

        fclose(imgListFile);
        return nFaces;
}

