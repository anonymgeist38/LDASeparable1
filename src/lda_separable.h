#ifndef LDA_SEPARABLE_H
#define LDA_SEPARABLE_H

#include <string>
#include <vector>
#include "opencv2/opencv.hpp"

namespace lda {

struct Data {
    Data();
    cv::Mat image;
    double label;
};

std::vector<Data> loadTrainingData(const std::string& path);
cv::Mat loadImage(const std::string& path);
cv::Mat getMean(const std::vector<Data>& training);
void centerData(std::vector<Data>& training, const cv::Mat& mean);
void compute(std::vector<Data>& data, int rho);
cv::Mat computeTemplate(int rho);
void setNumThreads(int numThreads);
void computeOnTestImages(const cv::Mat& visual,
                         const std::string& path,
                         const std::string& testFileResultsPath,
                         const std::string& visualizeW,
                         int rho,
                         int numThreads = 0);
void Visualize(const cv::Mat& matrix, const std::string& window);

} // namespace lda

#endif // LDA_SEPARABLE_H
