#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <direct.h>
#include <cmath>

using namespace std;
using namespace cv;

typedef unsigned char uchar;


template<typename T>
T my_max(T a, T b){
    if (a > b)
        return a;
    else
        return b;
}


uchar S(uchar value){
    if (value < 0)
        value = 0;
    else if(value > 255)
        value = 255;
    return value;
}


Mat mix(const Mat &im1, const Mat &im2, const double &u){
    int H = im1.rows, W = im1.cols, C = im1.channels();
    Mat g = Mat(H, W, CV_8UC3, Scalar(0));
    for (int h = 0; h < H; h++){
        for (int w = 0; w < W; w++){
            for (int c = 0; c < C; c++){
                g.at<Vec3b>(h, w)[c] = (1-u) * im1.at<Vec3b>(h, w)[c] + u * im2.at<Vec3b>(h, w)[c];
            }
        }
    }
    return g;
}

/*
// right down
Mat translation_rd(const Mat &im, const int &r, const int &d){
    Mat g(im.rows * 2, im.cols * 2, CV_8UC3, Scalar(0));
    int H = g.rows, W = g.cols, C = g.channels();

    for (int h = 0; h < H; h++){
        for(int w = 0; w < W; w++){
            for(int c = 0; c < C; c++){
                int u = w - r;
                int v = h - d;
                if(0 <= u && u < im.cols && 0 <= v && v < im.rows)
                    g.at<Vec3b>(h, w)[c] = im.at<Vec3b>(v, u)[c];
            }
        }
    }
    return g;
}
*/


// up
Mat translation_u(const Mat &im, const int up){
    Mat g(im.rows, im.cols, CV_8UC3, Scalar(0));
    int H = g.rows, W = g.cols, C = g.channels();

    for (int h = 0; h < H; h++){
        for(int w = 0; w < W; w++){
            for(int c = 0; c < C; c++){
                int v = h + up;
                if(0 <= v && v < im.rows)
                    g.at<Vec3b>(h, w)[c] = im.at<Vec3b>(v, w)[c];
            }
        }
    }
    return g;
}


Mat add(const Mat &im1, const Mat &im2){
    int H = im1.rows, W = im1.cols, C = im1.channels();
    Mat g(H, W, CV_8UC3, Scalar(0));
    for (int h = 0; h < H; h++){
        for (int w = 0; w < W; w++){
            for (int c = 0; c < C; c++){
                g.at<Vec3b>(h, w)[c] = S(im1.at<Vec3b>(h, w)[c] + im2.at<Vec3b>(h, w)[c]);
            }
        }
    }
    return g;
}


uchar bilinear(const Mat &im, double x, double y, double c){
    int i = x, j = y;
    double a = x - i, b = y - j;

    double up = (1-a) * im.at<Vec3b>(j, i)[c] + a * im.at<Vec3b>(j, i+1)[c];
    double down = (1-a) * im.at<Vec3b>(j+1, i)[c] + a * im.at<Vec3b>(j+1, i+1)[c];
    double value = (1-b) * up + b * down;
    return value;
}


// rotate around (w0, h0) in counterclockwise
Mat rotation(const Mat &im, const Mat &background, const double &angle, const int &w0, const int &h0){
    int H = im.rows, W = im.cols, C = im.channels();
    Mat g(H, W, CV_8UC3, Scalar(0));
    double radian = 3.1415927 * angle / 180.0;

    cout << g.cols << g.rows << g.channels() << endl;
    cout << background.cols << background.rows << background.channels() << endl;

    for (int h = 0; h < H; h++){
        for (int w = 0; w < W; w++){
            double u = (w-w0) * cos(-radian) + (h-h0) * sin(-radian) + w0;
            double v = -(w-w0) * sin(-radian) + (h-h0) * cos(-radian) + h0;
            for(int c = 0; c < C; c++){
                if(0 <= u && u < im.cols-1 && 0 <= v && v < im.rows-1){
                    g.at<Vec3b>(h, w)[c] = bilinear(im, u, v, c);
                }
                else{
                    g.at<Vec3b>(h, w)[c] = background.at<Vec3b>(h, w)[c];
                }
            }
        }
    }

    return g;
}


Mat cycle(const Mat &im, const Mat &background, double r, int w0, int h0){
    int H = im.rows, W = im.cols, C = im.channels();
    Mat g(H, W, CV_8UC3, Scalar(0));

    for (int h = 0; h < H; h++){
        for (int w = 0; w < W; w++){
            double d = (h - h0)*(h - h0) + (w - w0)*(w - w0);
            for(int c = 0; c < C; c++){
                if(d > r*r)
                    g.at<Vec3b>(h, w)[c] = im.at<Vec3b>(h, w)[c];
                else
                    g.at<Vec3b>(h, w)[c] = background.at<Vec3b>(h, w)[c];
            }
        }
    }
    return g;
}


Mat cat(const Mat &im1, const Mat &im2, const Mat &im3, const Mat &im4){
    int H1 = im1.rows, W1 = im1.cols;
    int H2 = im3.rows, W2 = im2.cols;
    int H = H1 + H2, W = W1 + W2, C = im1.channels();

    Mat g(H, W, CV_8UC3, Scalar(0));

    for (int h = 0; h < H; h++){
        for (int w = 0; w < W; w++) {
            for (int c = 0; c < C; c++) {
                if(h < H1)
                    if(w < W1)
                        g.at<Vec3b>(h, w)[c] = im1.at<Vec3b>(h, w)[c];
                    else
                        g.at<Vec3b>(h, w)[c] = im2.at<Vec3b>(h, w-W1)[c];
                else
                    if(w < W1)
                        g.at<Vec3b>(h, w)[c] = im3.at<Vec3b>(h-H1, w)[c];
                    else
                        g.at<Vec3b>(h, w)[c] = im4.at<Vec3b>(h-H1, w-W1)[c];
            }
        }
    }
    return g;
}


double filter(const Mat &im, int h, int w, int c){
    double result = 0;
    for(int y = h-1; y <= h+1; y++){
        for(int x = w-1; x <= w+1; x++){
            if(x == w && y == h)
                result += im.at<Vec3b>(y, x)[c] * 9;
            else{
                int u = x, v = y;
                if(u < 0) u = 0;
                else if(u >= im.cols) u = im.cols - 1;
                if (v < 0) v = 0;
                else if(v >= im.rows) v = im.rows - 1;
                result += -im.at<Vec3b>(v, u)[c];
            }
        }
    }
    return result;
}


// laplace sharpness
Mat sharp(const Mat &im){
    Mat g(im.rows, im.cols, CV_8UC3, Scalar(0));
    int H = g.rows, W = g.cols, C = g.channels();

    for (int h = 1; h < H-1; h++){
        for (int w = 1; w < W-1; w++){
            for(int c = 0; c < C; c++){
                g.at<Vec3b>(h, w)[c] = filter(im, h, w, c);
            }
        }
    }
    return g;
}


Mat zoom(const Mat &im, const double &h_scale, const double &w_scale){
    Mat g(im.rows * h_scale, im.cols * w_scale, CV_8UC3, Scalar(0));
    int H = g.rows, W = g.cols, C = g.channels();

    for (int h = 0; h < H; h++){
        for (int w = 0; w < W; w++){
            double v = h / h_scale;
            double u = w / w_scale;
            for(int c = 0; c < C; c++){
                if(u < im.cols-1 && v < im.rows-1)
                    g.at<Vec3b>(h, w)[c] = bilinear(im, u, v, c);
            }
        }
    }
//    g = sharp(g);
    return g;
}


int main()
{
    // load lr image
//    vector<string> scene = {"City_lr", "Forest_lr", "Snow mountain_lr"};
//    string dir = "./Landscapes/LR/";
//    Mat sub1_im1 = imread(dir + scene[0] + to_string(1) + ".jpeg", IMREAD_COLOR);
//    Mat sub1_im1_ = imread(dir + scene[0] + to_string(1) + ".jpeg", IMREAD_COLOR);
//    Mat sub1_im2 = imread(dir + scene[0] + to_string(2) + ".jpeg", IMREAD_COLOR);
//    Mat sub1_im3 = imread(dir + scene[0] + to_string(3) + ".jpeg", IMREAD_COLOR);
//    Mat sub1_im4 = imread(dir + scene[0] + to_string(4) + ".jpeg", IMREAD_COLOR);
//    Mat sub2_im1 = imread(dir + scene[1] + to_string(1) + ".jpeg", IMREAD_COLOR);
//    Mat sub2_im2 = imread(dir + scene[1] + to_string(2) + ".jpeg", IMREAD_COLOR);
//    Mat sub2_im3 = imread(dir + scene[1] + to_string(3) + ".jpeg", IMREAD_COLOR);
//    Mat sub2_im4 = imread(dir + scene[1] + to_string(4) + ".jpeg", IMREAD_COLOR);
//    Mat sub3_im1 = imread(dir + scene[2] + to_string(1) + ".jpeg", IMREAD_COLOR);
//    Mat sub3_im2 = imread(dir + scene[2] + to_string(2) + ".jpeg", IMREAD_COLOR);
//    Mat sub3_im3 = imread(dir + scene[2] + to_string(3) + ".jpeg", IMREAD_COLOR);
//    Mat sub3_im4 = imread(dir + scene[2] + to_string(4) + ".jpeg", IMREAD_COLOR);

    // TODO: 问题在于图片没读进来
    // load HR image
    vector<string> scene = {"City", "Forest", "Snow mountain"};
    string dir = "./Landscapes/";
    Mat sub1_im1 = imread(dir + scene[0] + "/" + scene[0] + to_string(1) + ".jpeg", IMREAD_COLOR);
//    Mat sub1_im1_ = imread(dir + scene[0] + "/" + scene[0] + to_string(1) + ".jpeg", IMREAD_COLOR);
    Mat sub1_im2 = imread(dir + scene[0] + "/" + scene[0] + to_string(2) + ".jpeg", IMREAD_COLOR);
    Mat sub1_im3 = imread(dir + scene[0] + "/" + scene[0] + to_string(3) + ".jpeg", IMREAD_COLOR);
    Mat sub1_im4 = imread(dir + scene[0] + "/" + scene[0] + to_string(4) + ".jpeg", IMREAD_COLOR);
    Mat sub2_im1 = imread(dir + scene[1] + "/" + scene[0] + to_string(1) + ".jpeg", IMREAD_COLOR);
    Mat sub2_im2 = imread(dir + scene[1] + "/" + scene[0] + to_string(2) + ".jpeg", IMREAD_COLOR);
    Mat sub2_im3 = imread(dir + scene[1] + "/" + scene[0] + to_string(3) + ".jpeg", IMREAD_COLOR);
    Mat sub2_im4 = imread(dir + scene[1] + "/" + scene[0] + to_string(4) + ".jpeg", IMREAD_COLOR);
    Mat sub3_im1 = imread(dir + scene[2] + "/" + scene[0] + to_string(1) + ".jpeg", IMREAD_COLOR);
    Mat sub3_im2 = imread(dir + scene[2] + "/" + scene[0] + to_string(2) + ".jpeg", IMREAD_COLOR);
    Mat sub3_im3 = imread(dir + scene[2] + "/" + scene[0] + to_string(3) + ".jpeg", IMREAD_COLOR);
    Mat sub3_im4 = imread(dir + scene[2] + "/" + scene[0] + to_string(4) + ".jpeg", IMREAD_COLOR);

    int frame = 60;
    int wait = 10;
    int idx = 0;
    mkdir("./result");

//    cout << sub1_im1.cols << endl;  // TODO
//    Mat im1(sub1_im2.rows * 2, sub1_im2.cols * 2, CV_8UC3, Scalar(0));
//    for (int f = 0; f <= frame; f++){
//        double a = 90.0 * f / double(frame);
//        Mat g1 = rotation(sub1_im1, sub2_im1, a, 0, 0);
//        Mat g2 = rotation(sub1_im2, sub2_im2, -a, sub1_im1.cols, 0);
//        Mat g3 = rotation(sub1_im3, sub2_im3, -a, 0, sub1_im1.rows);
//        Mat g4 = rotation(sub1_im4, sub2_im4, a, sub1_im1.cols, sub1_im1.rows);
//        im1 = cat(g1, g2, g3, g4);
//        if (f == 0 || f == frame)
//            for (int i = 0; i <= wait; i++)
//                imwrite("./result/image" + to_string(idx++) + ".jpeg", im1);
//        else
//            imwrite("./result/image" + to_string(idx++) + ".jpeg", im1);
//    }
//
//    Mat im2(sub2_im1.rows * 2, sub2_im1.cols * 2, CV_8UC3, Scalar(0));
//    int R = sqrt(sub2_im1.cols*sub2_im1.cols + sub2_im1.rows*sub2_im1.rows);
//    for (int f = 0; f <= frame; f++){
//        double r = R * f / double(frame);
//        Mat g1 = cycle(sub2_im1, sub3_im1, r, sub2_im1.cols, sub2_im1.rows);
//        Mat g2 = cycle(sub2_im2, sub3_im2, r, 0, sub2_im1.rows);
//        Mat g3 = cycle(sub2_im3, sub3_im3, r, sub2_im1.cols, 0);
//        Mat g4 = cycle(sub2_im4, sub3_im4, r, 0, 0);
//        im2 = cat(g1, g2, g3, g4);
//        if (f == frame)
//            for (int i = 0; i <= wait; i++)
//                imwrite("./result/image" + to_string(idx++) + ".jpeg", im2);
//        else
//            imwrite("./result/image" + to_string(idx++) + ".jpeg", im2);
//    }
//
//    Mat im3 = cat(sub1_im1, sub1_im2, sub1_im3, sub1_im4);
//    for (int f = 0; f <= frame; f++){
//        double u = f / double(frame);
//        Mat im3_mix = mix(im2, im3, u);
//        if (f == frame)
//            for (int i = 0; i <= wait; i++)
//                imwrite("./result/image" + to_string(idx++) + ".jpeg", im3_mix);
//        else
//            imwrite("./result/image" + to_string(idx++) + ".jpeg", im3_mix);
//    }
//
//    Mat im4 = cat(sub2_im1, sub2_im2, sub2_im3, sub2_im4);
//    for (int f = 0; f <= frame; f++){
//        int up = im4.rows * f / double(frame);
//        int down = im4.rows - up;
//        Mat im4_mix = add(translation_u(im3, up), translation_u(im4, -down));
//        if (f == frame)
//            for (int i = 0; i <= wait; i++)
//                imwrite("./result/image" + to_string(idx++) + ".jpeg", im4_mix);
//        else
//            imwrite("./result/image" + to_string(idx++) + ".jpeg", im4_mix);
//    }
//
//    Mat im5(sub3_im1.rows * 2, sub3_im1.cols * 2, CV_8UC3, Scalar(0));
//    for (int f = 0; f <= frame; f++){
//        int up = sub2_im1.rows * f / double(frame);
//        int down = sub2_im1.rows - up;
//        Mat g1 = add(translation_u(sub2_im1, up), translation_u(sub3_im1, -down));
//        Mat g2 = add(translation_u(sub2_im2, up), translation_u(sub3_im2, -down));
//        Mat g3 = add(translation_u(sub2_im3, -up), translation_u(sub3_im3, down));
//        Mat g4 = add(translation_u(sub2_im4, -up), translation_u(sub3_im4, down));
//        im5 = cat(g1, g2, g3, g4);
//        if (f == frame)
//            for (int i = 0; i <= wait; i++)
//                imwrite("./result/image" + to_string(idx++) + ".jpeg", im5);
//        else
//            imwrite("./result/image" + to_string(idx++) + ".jpeg", im5);
//    }
//
//    Mat im6(sub1_im1.rows * 2, sub1_im1.cols * 2, CV_8UC3, Scalar(0));
//    for (int f = 0; f <= frame; f++){
//        double scale = 1 + f / double(frame);
//        Mat g1 = zoom(sub3_im1, scale, scale);
//        Mat g2 = zoom(sub3_im2, scale, 2 - scale);
//        Mat g3 = zoom(sub3_im3, 2 - scale, scale);
//        Mat g4 = zoom(sub3_im4, 2 - scale,  2 - scale);
//        im6 = cat(g1, g2, g3, g4);
//        if (f == frame)
//            for (int i = 0; i <= wait; i++)
//                imwrite("./result/image" + to_string(idx++) + ".jpeg", im6);
//        else
//            imwrite("./result/image" + to_string(idx++) + ".jpeg", im6);
//    }

    Mat g = rotation(sub1_im1, sub2_im2, 45, 0, 0);
    imshow("g", g);

    waitKey(0);
    return 0;
}
