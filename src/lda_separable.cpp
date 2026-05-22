#include "lda_separable.h"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <cmath>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/core/parallel.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

namespace lda {

namespace {

Mat uVectors[9];
Mat vVectors[9];

Mat getTargetLabels(const vector<Data>& trainingData) {
    Mat labels(trainingData.size(), 1, CV_32FC1);
    for (size_t i = 0; i < trainingData.size(); i++) {
        labels.at<float>(static_cast<int>(i), 0) = static_cast<float>(trainingData[i].label);
    }
    return labels;
}

Mat computeLeastSquares(const Mat& contrax, const Mat& labels) {
    Mat contrax_trans = contrax.t();
    Mat tmp1 = (contrax_trans * contrax).inv();
    Mat u = tmp1 * contrax_trans * labels;
    return u;
}

Mat computeContractionInV(const vector<Data>& trainingData, const Mat& v) {
    Mat x_i = Mat::zeros(trainingData[0].image.rows, static_cast<int>(trainingData.size()), CV_32FC1);
    for (size_t ii = 0; ii < trainingData.size(); ii++) {
        Mat x_l = trainingData[ii].image * v;
        x_l.col(0).copyTo(x_i.col(static_cast<int>(ii)));
    }
    return x_i.t();
}

Mat computeContractionInU(const vector<Data>& trainingData, const Mat& u) {
    Mat x_j = Mat::zeros(static_cast<int>(trainingData.size()), trainingData[0].image.cols, CV_32FC1);
    Mat u_t = u.t();
    for (size_t ii = 0; ii < trainingData.size(); ii++) {
        Mat x_l = u_t * trainingData[ii].image;
        x_l.row(0).copyTo(x_j.row(static_cast<int>(ii)));
    }
    return x_j;
}

Mat getRandomVector(int dim_vector) {
    RNG rng_obj;
    rng_obj.state = getTickCount();
    Mat vec(dim_vector, 1, CV_32FC1);
    for (int i = 0; i < vec.rows; i++) {
        vec.at<float>(i, 0) = rng_obj.uniform(0.0, 1.0);
    }
    return vec;
}

double getMagnitude(const Mat& vec) {
    double result = 0.0;
    for (int i = 0; i < vec.rows; i++) {
        double value = vec.at<float>(i, 0);
        result += value * value;
    }
    return sqrt(result);
}

void orthogonalize(Mat vectors[9], short index) {
    Mat result = vectors[index].clone();
    for (short i = 0; i < index; i++) {
        result -= (vectors[index].dot(vectors[i])) * vectors[i];
    }
    result /= getMagnitude(result);
    vectors[index] = result;
}

void computeForRho1(vector<Data>& data, int index) {
    Mat u = getRandomVector(data[0].image.rows);
    Mat Y = getTargetLabels(data);
    Mat uT, v, diff;
    do {
        Mat x_j = computeContractionInU(data, u);
        v = computeLeastSquares(x_j, Y);
        Mat x_i = computeContractionInV(data, v);
        uT = u;
        u = computeLeastSquares(x_i, Y);
        diff = u - uT;
    } while (getMagnitude(diff) > 0.0001);
    uVectors[index] = u;
    vVectors[index] = v;
    orthogonalize(uVectors, static_cast<short>(index));
    orthogonalize(vVectors, static_cast<short>(index));
}

Mat computeClassMean(const vector<Data>& data, double label) {
    int num_occ = 0;
    Mat class_mean = Mat::zeros(data[0].image.rows, data[0].image.cols, data[0].image.type());
    for (size_t i = 0; i < data.size(); i++) {
        if (data[i].label == label) {
            class_mean += data[i].image;
            num_occ++;
        }
    }
    class_mean /= num_occ;
    return class_mean;
}

} // anonymous namespace

Data::Data() : image(), label(0.0) {}

vector<Data> loadTrainingData(const string& path) {
    vector<Data> trainingData;
    stringstream sstream;
    string prefix = "uiucPos-81-31-";
    for (short i = 0; i < 124; i++) {
        sstream.str("");
        sstream.clear();
        sstream << path << prefix << setfill('0') << setw(2) << i << ".pgm";
        Data d;
        d.image = loadImage(sstream.str());
        d.label = +1.0 / 124.0;
        trainingData.push_back(d);
    }
    prefix = "uiucNeg-81-31-";
    for (short i = 0; i < 2442; i++) {
        sstream.str("");
        sstream.clear();
        sstream << path << prefix << setfill('0') << setw(2) << i << ".pgm";
        Data d;
        d.image = loadImage(sstream.str());
        d.label = -1.0 / 2442.0;
        trainingData.push_back(d);
    }
    return trainingData;
}

Mat loadImage(const string& path) {
    Mat image = imread(path.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
    image.convertTo(image, CV_32FC1);
    return image;
}

Mat getMean(const vector<Data>& training) {
    Mat mean = Mat::zeros(training[0].image.size(), training[0].image.type());
    for (size_t i = 0; i < training.size(); i++) {
        mean += training[i].image;
    }
    mean /= static_cast<double>(training.size());
    return mean;
}

void centerData(vector<Data>& training, const Mat& mean) {
    for (size_t i = 0; i < training.size(); i++) {
        training[i].image -= mean;
    }
}

void compute(vector<Data>& data, int rho) {
    for (int i = 0; i < rho; i++) {
        computeForRho1(data, i);
    }
}

Mat computeTemplate(int rho) {
    Mat fTemplate = Mat::zeros(uVectors[0].rows, vVectors[0].rows, CV_32FC1);
    for (int i = 0; i < rho; i++) {
        fTemplate += uVectors[i] * vVectors[i].t();
    }
    GaussianBlur(fTemplate, fTemplate, Size(0, 0), 1.5);
    return fTemplate;
}

void setNumThreads(int numThreads) {
    if (numThreads > 0) {
        cv::setNumThreads(numThreads);
    } else {
        cv::setNumThreads(0);
    }
}

void computeOnTestImages(const Mat& visual,
                         const string& path,
                         const string& testFileResultsPath,
                         const string& visualizeW,
                         int rho,
                         int numThreads) {
    Mat normalizedVisual;
    normalize(visual, normalizedVisual, 0, 255, NORM_MINMAX, CV_8UC1);

    vector<int> compressionParams;
    compressionParams.push_back(CV_IMWRITE_JPEG_QUALITY);
    compressionParams.push_back(95);

    Mat templ = normalizedVisual;
    templ.convertTo(templ, CV_32FC1);

    string prefix = "search";
    const int numTestImages = 170;

    parallel_for_(Range(0, numTestImages), [&](const Range& range) {
        for (int i = range.start; i < range.end; ++i) {
        Mat img = imread(format("%s%d.PGM", path.c_str(), i), CV_LOAD_IMAGE_GRAYSCALE);
        img.convertTo(img, CV_32FC1);
        int result_cols = img.cols - templ.cols + 1;
        int result_rows = img.rows - templ.rows + 1;
        Mat result;
        result.create(result_rows, result_cols, CV_32FC1);
        matchTemplate(img, templ, result, CV_TM_CCOEFF);

        Mat general_mask = Mat::ones(result.rows, result.cols, CV_8UC1);
        for (int k = 0; k < 5; k++) {
            minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, general_mask);
            result.at<float>(minLoc) = 1.0f;
            result.at<float>(maxLoc) = 0.0f;
            matchLoc = maxLoc;

            float k_overlapping = 1.f;
            int template_w = ceil(k_overlapping * templ.cols);
            int template_h = ceil(k_overlapping * templ.rows);
            int x = matchLoc.x - template_w / 2;
            int y = matchLoc.y - template_h / 2;
            if (y < 0) y = 0;
            if (x < 0) x = 0;
            if (template_w + x > general_mask.cols) template_w = general_mask.cols - x;
            if (template_h + y > general_mask.rows) template_h = general_mask.rows - y;
            Mat template_mask = Mat::zeros(template_h, template_w, CV_8UC1);
            template_mask.copyTo(general_mask(Rect(x, y, template_w, template_h)));

            Mat img_display = img.clone();
            cvtColor(img_display, img_display, CV_GRAY2BGR);
            rectangle(img_display, matchLoc,
                      Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows),
                      Scalar(0, 255, 0), 2, 8, 0);

            imwrite(format("%s%d%d.jpg", (testFileResultsPath + to_string(i) + "_" + prefix + "_" + to_string(k)).c_str(), i, k),
                    img_display, compressionParams);
        }
    }
    cout << "Done...Check the folder uiucTestResults Folder for projector on test images. " << endl;
}

void Visualize(const Mat& matrix, const string& window) {
    vector<int> compressionParams;
    compressionParams.push_back(CV_IMWRITE_JPEG_QUALITY);
    compressionParams.push_back(95);
    int rho = 9;

    Mat output;
    normalize(matrix, output, 0, 255, NORM_MINMAX, CV_8UC1);
    if (window == "Visualize projector matrix") {
        string path = "images/Visualizing_tensor_projection_";
        imwrite(format("%s%d.jpg", path.c_str(), rho), output, compressionParams);
    }
    imshow(window, output);
    waitKey(0);
}

} // namespace lda
