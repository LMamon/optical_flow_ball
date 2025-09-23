#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    //open cam
    VideoCapture cap(0);
    
    if (!cap.isOpened()) {
        cerr << "Error: Could not open webcam\n";
        return -1;
    }
    
    Mat frame;
    cap >> frame;
    Mat mask = Mat::zeros(frame.size(), frame.type());
    
    if (frame.empty()) {
        cerr << "Error: Blank initial frame\n";
        return -1;
    }

    Mat grayPrev;
    cvtColor(frame, grayPrev, COLOR_BGR2GRAY);
    
    //ball state
    Point2f ball(frame.cols / 2, frame.rows /3 );
    Point2f velocity(0, 1);
    float radius = 50;
    float gravity = 0.1f;

    const int MAX_CORNERS = 200;

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        Mat grayCurr;
        cvtColor(frame, grayCurr, COLOR_BGR2GRAY);
        
        //physics
        velocity.y += gravity;
        ball += velocity;

        //bounce off window
        if (ball.y + radius >= frame.rows) {
            ball.y = frame.rows - radius;
            velocity.y = -abs(velocity.y) * 0.8f;
        }
        if (ball.y - radius <= 0) {
            ball.y = radius;
            velocity.y = abs(velocity.y);
        }
        
        if ((ball.x - radius) <= 0 || (ball.x + radius) >= frame.cols) {
            velocity.x = -velocity.x * 0.8f;
            ball.x = std::clamp(ball.x, (float)radius, (float)(frame.cols - radius));
        }

        //ROI
        int roiSize = radius * 3;
        int x = max(0, (int)ball.x - roiSize / 2);
        int y = max(0, (int)ball.y - roiSize / 2);
        int w = min(roiSize, frame.cols - x);
        int h = min(roiSize, frame.rows - y);

        Rect roi(x, y, w, h);

        //mask for reseeding
        Mat maskROI = Mat::zeros(grayCurr.size(), CV_8UC1);
        maskROI(roi).setTo(255);

        //feature detection
        vector<Point2f> pointsPrev, pointsNext;
        goodFeaturesToTrack(grayPrev, pointsPrev, MAX_CORNERS,
                                0.2,
                                7,
                                maskROI,
                                7, 
                                false, 
                                0.04);

        if (!pointsPrev.empty()) {
            vector<uchar> status;
            vector<float> err;

            calcOpticalFlowPyrLK(grayPrev, 
                                    grayCurr, 
                                    pointsPrev, 
                                    pointsNext,
                                    status, 
                                    err);

            Point2f avgMotion(0, 0);
            int count = 0;

            for (size_t i = 0; i < pointsNext.size(); i++) { //update fp in ROI
                if (status[i]) {
                    avgMotion += (pointsNext[i] - pointsPrev[i]);
                    count++;
                }
            }

            if (count>0 ) {
                avgMotion.x /= count;
                avgMotion.y /= count;
                velocity.x += avgMotion.x * 0.5f;
                velocity.y += avgMotion.y * 0.5f;
            }
        }

        circle(frame, ball, radius, Scalar(0, 0, 255), -1);
        rectangle(frame, roi, Scalar(0, 255, 0), 2); //roi where points are tracked

        imshow("optical flow ball", frame);

        grayPrev = grayCurr.clone();

        char key = (char) waitKey(30);
        if(key == 27 || key == 'q')
            break;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}