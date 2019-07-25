#include <iostream>
#include <vector>
#include <time.h>
#include <math.h>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

#define Linear          0
#define Sigmoid         1
#define ReLU            2

// IMAGE PADDING
Mat padding(Mat image, int padding_size){
    clock_t begin = clock();
    // padding size : image + padding x 2
    Mat result(image.rows+padding_size*2, image.cols+padding_size*2, image.type());

    for (int i=0; i<result.rows; i++){
        for (int j=0; j<result.cols; j++){
            // padding
            if(i<padding_size || i > result.rows - padding_size - 1 
            || j < padding_size || j > result.cols - padding_size -1){
                    result.at<uchar>(i,j) = 0;
            }   
            // original image
            else    result.at<uchar>(i,j) = image.at<uchar>(i-padding_size,j-padding_size);
        }
    }
    clock_t end = clock();
    cout << "padding time:\t" << double(end-begin)/CLOCKS_PER_SEC << endl;
    return result;
}

// IMAGE CONVOLUTION by one filter
// padding size recommend filter size / 2, stride recommend 1
Mat conv2D(Mat image, Mat filter,int padding_size, int stride=1){
    clock_t begin = clock();
    Mat pad = padding(image,padding_size);
    Mat result((pad.rows-filter.rows)/stride + 1, (pad.cols-filter.cols)/stride + 1, image.type());                 

    for(int i=0; i<result.rows; i++){
        for(int j=0; j<result.cols; j++){
            double temp = 0;
            for(int k=0; k<filter.rows; k++){     // convolution
                for(int l=0; l<filter.cols; l++){
                        temp += pad.at<uchar>(i*stride+k,j*stride+l) * filter.at<double>(k,l);           
                }
            }
            temp = round(temp);
            if(temp<0)          temp = 0;
            else if(temp>255)   temp = 255;
            result.at<uchar>(i,j) = temp;
        }
    }
    
    clock_t end = clock();
    cout << "convolution time:\t" << double(end-begin)/CLOCKS_PER_SEC << endl;
    return result;
}

// MAX POOLING
Mat max_pooling(Mat image, int pooling_size=2, int stride=2){
    clock_t begin = clock();
    Mat result((image.rows-pooling_size)/stride + 1, (image.cols-pooling_size)/stride + 1, image.type());
    
    for(int i=0; i<result.rows; i++){
        for(int j=0; j<result.cols; j++){           
            result.at<uchar>(i,j) = 0;              // Init

            for(int k=0; k<pooling_size; k++){      // convolution
                for(int l=0; l<pooling_size; l++){
                    if(result.at<uchar>(i,j)<image.at<uchar>(i*stride + k,j*stride + l))
                        result.at<uchar>(i,j) = image.at<uchar>(i*stride + k,j*stride + l);            
                }
            }

        }
    }
    
    clock_t end = clock();
    cout << "max pooling time:\t" << double(end-begin)/CLOCKS_PER_SEC << endl;
    return result;
}

// AVERAGE POOLING
Mat average_pooling(Mat image, int pooling_size=2, int stride=2){
    clock_t begin = clock();
    Mat result((image.rows-pooling_size)/stride + 1, (image.cols-pooling_size)/stride + 1, image.type());
    
    for(int i=0; i<result.rows; i++){
        for(int j=0; j<result.cols; j++){
            int temp = 0;

            for(int k=0; k<pooling_size; k++){     // convolution
                for(int l=0; l<pooling_size; l++){
                        temp += image.at<uchar>(i*stride + k, j*stride + l);
                }
            }
            result.at<uchar>(i,j) = temp/(pooling_size*pooling_size);       
            delete(temp);
        }
    }

    clock_t end = clock();
    cout << "average pooling time:\t" << double(end-begin)/CLOCKS_PER_SEC << endl;
    return result;
}

double activation(double sigma, int type=Linear){
    switch(type){
        case(Linear):   return sigma;
        case(Sigmoid):  return 1./(1-exp(-sigma));
        case(ReLU):     if (sigma<0)    return 0;
                        else            return sigma;
    } 
}

Mat activation(Mat image, int type=Linear){
    clock_t begin = clock();
    Mat result(image.rows, image.cols, image.type());
    
    for(int i=0; i<result.rows; i++){
        for(int j=0; j<result.cols; j++){
            result.at<uchar>(i,j) = activation(image.at<uchar>(i,j),type);
        }
    }
    
    clock_t end = clock();
    cout << "activation time:\t" << double(end-begin)/CLOCKS_PER_SEC << endl;
    return result;
}

void comp(Mat img1, Mat img2, int print=0){
    if (img1.rows != img2.rows || img1.cols != img2.cols){
        cout << "image1 row:\t" << img1.rows << "\timage2 row" << img2.rows << endl;
        cout << "image1 col:\t" << img1.cols << "\timage2 col" << img2.cols << endl;
        cout << "IMAGE SIZE NOT SAME" << endl;
        return;
    }
    int cnt = 0;
    for (int i=0; i<img1.rows; i++){
        for (int j=0; j<img1.cols; j++){
            for (int ch=0; ch<3; ch++){
                if (abs(img1.at<double>(i,j)[ch] - img2.at<double>(i,j)[ch])>0){
                    //if(i!=0&&i!=img1.rows-1&&j!=0&&j!=img1.cols-1){
                    if (print){
                        cout << "location\n";
                        cout << i << "\t" << j << "\tch:" << ch << endl;
                        cout << "img1 info\t";
                        cout << (int)img1.at<double>(i,j)[ch] << endl;
                        cout << "img2 info\t";
                        cout << (int)img2.at<double>(i,j)[ch] << endl << endl;
                    }//}
                    //if(i!=0&&i!=img1.rows-1&&j!=0&&j!=img1.cols-1)
                        cnt++;
                }
            }
        }
    }
    cout << cnt << endl;
    return;
}