#include <iostream>
#include <cstdlib>
#include <string>
#include <ctime>
#include<cmath>

#include <vector>

#include <opencv2\opencv.hpp>

#include <opencv2\nonfree\features2d.hpp>
#include <opencv2\nonfree\nonfree.hpp>
#include <opencv2\legacy\legacy.hpp>

#include "dirent.h"

#define PI 3.1415926f 

using namespace std;
using namespace cv;

string convertInt(int number)
{
   stringstream ss;//create a stringstream
   ss << number;//add number to the stream
   return ss.str();//return a string with the contents of the stream
}
//*
double CCW( Point2f a, Point2f b , Point2f c)
{
    Point2f z;
    z.x = b.x - a.x;
    z.y = b.y - a.y;
    Point2f x;
    x.x = c.x - a.x;
    x.y = c.y - a.y;

    double res = (z.x * x.y - z.y * x.x);
    return res;   
}
bool falsePositive( vector<Point2f> p ) {
    int distThresh = 20;
                
    for(int i=1;i<3;i++){
        Point2f a = Point2f(p[i+1].x - p[i].x,p[i+1].y - p[i].y);
        Point2f b = Point2f(p[i-1].x - p[i].x,p[i-1].y - p[i].y);
        double dp = (a.x * b.x + a.y * b.y);
        double am = sqrt(a.x * a.x + a.y * a.y);
        double bm = sqrt(b.x * b.x + b.y * b.y);
        double angle = dp / (am * bm);
        angle = acos(angle) * 180 / PI;
        if (angle > 150 || angle < 30)
        {
            return true;
        }
    }

    for( int i=0;i<4;i++){
        for( int j=i+1;j<4;j++){
            if( sqrt ( pow(abs(p[i].x - p[j].x),2) - pow(abs(p[i].y - p[j].y),2) )  < distThresh){
                    return true;
            }
        }
    }

    if (CCW(p[0], p[1], p[2])*CCW(p[1], p[2], p[3])<0 || CCW(p[3], p[0], p[1]) * CCW(p[0], p[1], p[2]) < 0)
    {
        return true;
    }



    return false;
}


vector< vector<cv::KeyPoint> > modelKeyPoints;
vector< cv::Mat > modelDescriptors;
void Precompute(vector<string> files){
	// TIMER
	clock_t start;
    double duration;
	start = std::clock();

	cv::SiftFeatureDetector detector;
	cv::SiftDescriptorExtractor extractor;
	Mat source;
	int l = files.size();
	for( int i(0);i<l;i++ ){
		source = imread(files[i].c_str(), CV_LOAD_IMAGE_GRAYSCALE);  
		std::vector<cv::KeyPoint> keypoints;
		detector.detect(source, keypoints);
		modelKeyPoints.push_back( keypoints );

		Mat descriptor;
		extractor.compute(source , keypoints , descriptor );
		modelDescriptors.push_back(descriptor);
	}
	
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
	cout << "PRECOMPUTING : " << duration << endl;
}

Mat Search( Mat source , vector<string> files ){
	//TIMER
	clock_t start;
    double duration;
	start = std::clock();


	cv::Mat output;
	output = source;
	//cv::FastFeatureDetector detector;
	cv::SiftFeatureDetector detector;
	cv::SiftDescriptorExtractor extractor;
	// Keypoints
	std::vector<cv::KeyPoint> keypoints;
	detector.detect(source, keypoints);
	// Descriptor
	Mat descriptor;
	extractor.compute(source , keypoints , descriptor );


	duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
	cout << "FRAME PROCESSING : " << duration << endl;
	start = std::clock();
	//Declaring flann based matcher
	FlannBasedMatcher matcher;
	float nndrRatio = 0.7f;
	
	int l = files.size();
	for( int ioi(0);ioi<l;ioi++ ){
		
		vector< vector< DMatch >  > matches;
		matcher.knnMatch( modelDescriptors[ioi], descriptor, matches, 2 ); // find the 2 nearest neighbors
		
		vector< DMatch > good_matches;
		good_matches.reserve(matches.size());  
   
		for (size_t i = 0; i < matches.size(); ++i)
		{ 
			if (matches[i].size() < 2)
						continue;
   
			const DMatch &m1 = matches[i][0];
			const DMatch &m2 = matches[i][1];
        
			if(m1.distance <= nndrRatio * m2.distance)        
			good_matches.push_back(m1);     
		}
		if ( good_matches.size() > 4 ) {
		    vector< Point2f >  obj;
			vector< Point2f >  scene;
		
			for( unsigned int i = 0; i < good_matches.size(); i++ )
			{
				//-- Get the keypoints from the good matches
				obj.push_back( modelKeyPoints[ioi][ good_matches[i].queryIdx ].pt );
				scene.push_back( keypoints[ good_matches[i].trainIdx ].pt );
			}
 
			Mat H = findHomography( obj, scene, CV_RANSAC );
  
			//-- Get the corners from the image_1 ( the object to be "detected" )
			std::vector< Point2f > obj_corners(4);
			obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( modelDescriptors[ioi].cols, 0 );
			obj_corners[2] = cvPoint( modelDescriptors[ioi].cols, modelDescriptors[ioi].rows ); obj_corners[3] = cvPoint( 0, modelDescriptors[ioi].rows );
			std::vector< Point2f > scene_corners(4);
	
			perspectiveTransform( obj_corners, scene_corners, H);
			if( falsePositive(scene_corners) )continue;
			Scalar color = cv::Scalar(0.5,0.5,0.0);
			//-- Draw lines between the corners (the mapped object in the scene - image_2 ) 
			line( output, scene_corners[0] , scene_corners[1], color, 2 ); //TOP line
			line( output, scene_corners[1] , scene_corners[2], color, 2 );
			line( output, scene_corners[2] , scene_corners[3], color, 2 );
			line( output, scene_corners[3] , scene_corners[0] , color, 2 ); 
		}
	}
	
	duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
	cout << "MATCHING : " << duration << endl;

	return output;
}

int main(int argc ,char *argv[] ){
    
	//string as = "1.jpg";
	
	if ( argc < 3 )return 0;
	cout << argc << endl << argv[1] << endl << argv[2] << endl;
	vector<string> filenames;
	string s( argv[1] );

	filenames.push_back( s );
	Precompute(filenames);

	cvNamedWindow("RES");
	
	Mat img , res;
	string scene(argv[2]);
	
	img = imread(scene);
		
	res = Search(img, filenames);
	imshow("RES" , res );
	cvWaitKey(0);
	
	cvDestroyWindow( "RES" );
	return 0;
}