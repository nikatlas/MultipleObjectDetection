#include <iostream>
#include <cstdlib>
#include<ctime>
#include <string>

#include <opencv2\opencv.hpp>
#include <opencv\cv.h>
#include <opencv2\highgui\highgui.hpp>

using namespace std;
using namespace cv;

string convertInt(int number)
{
   stringstream ss;//create a stringstream
   ss << number;//add number to the stream
   return ss.str();//return a string with the contents of the stream
}

Point p1,p2;
bool flag = true;
bool crop = false;
void on_mouse( int event, int x, int y, int flags, void* param ){
	 if (event == 1)
     {
		 //cout << x << "  " << y << endl;
		 if( flag ){
			 p1.x = x;
			 p1.y = y;
			 flag = false;
		 }
		 else{
			 p2.x = x;
			 p2.y = y;
			 crop = true;
			 flag = true;
		 }
	 }
	
}

int main(){
	
	srand((unsigned)time(0)); 
    CvCapture* capture = cvCaptureFromCAM( CV_CAP_ANY );
    cvNamedWindow("Camera");
	
	IplImage* frame = cvQueryFrame( capture );
	Mat img;
	cout << "STARTING ---- " << endl;
	
    while (true)
	{
        frame = cvQueryFrame(capture);
        img = frame;

		imshow("Camera", img );
		if ( cvWaitKey(5) == 'c')
        {
			cvSetMouseCallback("Camera" , on_mouse , 0 );
			cout << "CROP IT -------- S to save" << endl;
			while( cvWaitKey(1)!='s' ){
				if( crop ){
					Rect myROI(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
					Mat cimg = img(myROI);
					imshow("Camera" , cimg);
					crop = false;
					img = cimg;
				}
			}
			int r = rand()*rand()%1000000;
			cout << "images/frame_"+convertInt(r)+".jpg" <<endl;
			imwrite( "images/frame_"+convertInt(r)+".jpg", img );
            break;
        }
    }
    
	


	cvReleaseCapture( &capture );
    cvDestroyWindow( "Camera" );
	return 0;
}