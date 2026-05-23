/*
 * main.cpp
 *
 *  Created on: Jul 18, 2015
 *      Author: raghu
 */

/*
 * main.cpp
 *
 *  Created on: Jul 5, 2015
 *      Author: raghu
 */
#define DEBUG
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

#include <vector>
#include <string>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/mat.hpp"
#include "video_processor.h"

#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/stream.hpp>

using namespace boost::iostreams;

#define NUM_OF_IMAGES 2566

using namespace std;
using namespace cv;
Mat uVectors[9];
Mat vVectors[9];
class Data
{
public:
	Data(){label = 0.0;}
	Mat image;
	double label;
};
void Visualize(Mat matrix, string window);

void computeOnTestImages(Mat visual, string& path, string& testFileResultsPath,
		string& visualizeW, int rho) {
	// Perform template matching on test set.

	normalize(visual, visual, 0, 255, NORM_MINMAX, CV_8UC1);

	vector<int> compressionParams;
	compressionParams.push_back(cv::IMWRITE_JPEG_QUALITY);
	compressionParams.push_back(95);

/*
	imwrite(format("%s%d.PGM", (visualizeW).c_str(), rho), visual,
			compressionParams);
*/

	vector<Mat> results;
	Mat result;

	Mat img;

	Mat templ = visual;

	templ.convertTo(templ, CV_32FC1);

	Mat img_display;
	string responseMapsPath = "images/responseMaps/Result_";

	double minVal;
	double maxVal;
	Point minLoc;
	Point maxLoc;
	Point matchLoc;
	string prefix = "search";

	// Apply on test images
	for (int i = 0; i < 170; i++) {

		img = imread(format("%s%d.PGM", (path).c_str(), i),
				cv::IMREAD_GRAYSCALE);
		if (img.empty()) {
			cerr << "Skipping missing or unreadable image: " << format("%s%d.PGM", (path).c_str(), i) << endl;
			continue;
		}

		img.convertTo(img, CV_32FC1);

		int result_cols = img.cols - templ.cols + 1;
		int result_rows = img.rows - templ.rows + 1;
		if (result_cols <= 0 || result_rows <= 0) {
			cerr << "Skipping image with invalid result size: " << format("%s%d.PGM", (path).c_str(), i)
			     << " img=" << img.cols << "x" << img.rows
			     << " templ=" << templ.cols << "x" << templ.rows << endl;
			continue;
		}

		result.create(result_rows, result_cols, CV_32FC1); //CV_32FC1

		matchTemplate(img, templ, result, cv::TM_CCOEFF);
//		normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat() );

		//MatchTemplate returns a similarity map and not a location.s
		//Do the matching and normalize.

		Mat general_mask = Mat::ones(result.rows, result.cols, CV_8UC1);

		for (int k = 0; k < 5; k++) {

			minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, general_mask); //general_mask

			if (minLoc.x < 0 || minLoc.y < 0 || minLoc.x >= result.cols || minLoc.y >= result.rows ||
			    maxLoc.x < 0 || maxLoc.y < 0 || maxLoc.x >= result.cols || maxLoc.y >= result.rows) {
				cerr << "Invalid match location for image " << format("%s%d.PGM", (path).c_str(), i)
				     << " minLoc=" << minLoc << " maxLoc=" << maxLoc << " result=" << result_rows << "x" << result_cols << endl;
				continue;
			}

			//just to visually observe centering I stay this part of code:


			result.at<float>(minLoc) = 1.0; //
			result.at<float>(maxLoc) = 0.0; //

			matchLoc = maxLoc;

			float k_overlapping = 1.f;	//little overlapping is good for my task


			//create template size for masking objects, which have been found,
			//to be excluded in the next loop run


			int template_w = ceil(k_overlapping * templ.cols);
			int template_h = ceil(k_overlapping * templ.rows);

			int x = matchLoc.x - template_w / 2;
			int y = matchLoc.y - template_h / 2;

			//shrink template-mask size to avoid boundary violation
			if (y < 0)
				y = 0;
			if (x < 0)
				x = 0;
			//will template come beyond the mask?:if yes-cut off margin;
			if (template_w + x > general_mask.cols)
				template_w = general_mask.cols - x;

			if (template_h + y > general_mask.rows)
				template_h = general_mask.rows - y;


			//set the negative mask to prevent repeating

			Mat template_mask = Mat::zeros(template_h, template_w, CV_8UC1);

			template_mask.copyTo(
					general_mask(cv::Rect(x, y, template_w, template_h)));


			Mat img_display = img.clone();
			cvtColor(img_display, img_display, cv::COLOR_GRAY2BGR);

			rectangle(img_display, matchLoc,
					Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows),
					Scalar(0, 255, 0), 2, 8, 0);

			rectangle(result, matchLoc,
					Point(matchLoc.x - templ.cols, matchLoc.y - templ.rows),
					Scalar(0, 255, 0), 2, 8, 0);

			//Visualize(img_display, to_string(k));
			imwrite(format("%s%d%d.jpg", (testFileResultsPath+to_string(i)+"_"+prefix+"_"+to_string(k)).c_str(), i, k), img_display, compressionParams);
		}

	}
	cout<<"Done...Check the folder uiucTestResults Folder for projector on test images. "<<endl;
}

Mat getTargetLabels(vector<Data>& trainingData){

	Mat labels(trainingData.size(), 1, CV_32FC1);

	for(int i = 0; i < trainingData.size(); i++){

		labels.at<float>(i,0) = trainingData[i].label;
	}
	return labels;
}

Mat computeLeastSquares(Mat& contrax, Mat& labels){
	Mat contrax_trans = contrax.t();
	Mat tmp1 = (contrax_trans * contrax).inv();

	Mat u = tmp1 * contrax_trans * labels;

	return u;
}

Mat computeContractionInV(vector<Data>& trainingData, Mat& v){
	Mat x_i = Mat::zeros(trainingData[0].image.rows, trainingData.size(), CV_32FC1);

	for(int ii = 0; ii < trainingData.size(); ii++){

		Mat x_l = trainingData[ii].image * v;
		x_l.col(0).copyTo(x_i.col(ii));
	}

	return x_i.t();
}
Mat computeContractionInU(vector<Data>& trainingData, Mat& u){


	Mat x_j = Mat::zeros(trainingData.size(), trainingData[0].image.cols, CV_32FC1);

	Mat u_t = u.t();

	for(int ii = 0; ii < trainingData.size(); ii++){

		Mat x_l = u_t * trainingData[ii].image;

		x_l.row(0).copyTo(x_j.row(ii));
	}

	return x_j;
}
Mat getRandomVector(int dim_vector){
	RNG rng_obj;
	rng_obj.state = cv::getTickCount();

	Mat vec(dim_vector, 1, CV_32FC1);
	for(int i = 0; i < vec.rows; i++){
		vec.at<float>(i,0) = rng_obj.uniform(0., 1.);
	}
	return vec;
}
double getMagnitude(Mat &vec)
{
	double result = 0.0;
	for(short int i=0; i<vec.rows; i++)
	{
		result += (vec.at<float>(i,0) * vec.at<float>(i,0));
	}
	return sqrt(result);
}
void orthoganalize(Mat vectors[9], short int index)
{
	Mat result = vectors[index].clone();

	for(short int i=0; i<index; i++)
	{
		result -= (vectors[index].dot(vectors[i])) * vectors[i];
	}
	result /= getMagnitude(result);

	vectors[index] = result;
}

void computeFor_rho_1(vector<Data>& data, int index){

	Mat u = getRandomVector(data[0].image.rows);
	//Mat v = getRandomVector(data[0].image.cols);
	Mat Y = getTargetLabels(data);

	Mat uT,v, diff;

	do{

		Mat x_j = computeContractionInU(data, u);

		v = computeLeastSquares(x_j, Y);

		Mat x_i = computeContractionInV(data, v);

		uT = u;
		u = computeLeastSquares(x_i, Y);

		diff = u - uT;

	}while(getMagnitude(diff) > 0.0001);

	uVectors[index] = u;
	vVectors[index] = v;

	orthoganalize(uVectors, index);
	orthoganalize(vVectors, index);
}

void compute(vector<Data>& data, int rho)
{
	for(int i=0;i<rho;i++)
	{
		computeFor_rho_1(data, i);
	}
}

Mat computeClassMean(vector<Data>& data, double label){
	int num_occ = 0;
	Mat class_mean = Mat::zeros(data[0].image.rows, data[0].image.cols, data[0].image.type());
	for(int i = 0; i < data.size(); i++){
		if(data[i].label==label){
			class_mean += data[i].image;
			num_occ++;
		}
	}
	class_mean /= num_occ;
	return class_mean;
}

Mat computeTemplate(int rho)
{
	Mat fTemplate = Mat::zeros(uVectors[0].rows, vVectors[0].rows, CV_32FC1);
	for(int i=0;i<rho;i++)
	{
		fTemplate += uVectors[i] * vVectors[i].t();
	}
	GaussianBlur(fTemplate, fTemplate, Size(0,0), 1.5);
	return fTemplate;
}

void centerData(vector<Data> &training, Mat mean){
	for(int i = 0; i < training.size(); i++){
		training[i].image -= mean;
	}
}

Mat getMean(vector<Data> &training)
{
	Mat mean = cv::Mat::zeros(training[0].image.size(), training[0].image.type());

	for(short int i=0;i<training.size();i++)
	{
		mean += training[i].image;
	}

	mean /= training.size();
	return mean;
}

Mat LoadImage(string path)
{
	Mat image = imread(path.c_str(), cv::IMREAD_GRAYSCALE);
	image.convertTo(image, CV_32FC1);
	return image;
}

vector<Data> loadTrainingData(string path)
{
	vector<Data> trainingData;
	stringstream sstream;
	string prefix = "uiucPos-81-31-";
	for(short int i=0;i<124;i++)
	{
		sstream.str("");
		sstream<<path<<prefix<<setfill('0')<<setw(2)<<i<<".pgm";

		Data d;
		d.image = LoadImage(sstream.str());
		d.label = +1.0 / 124.0;
		trainingData.push_back(d);
	}
	prefix = "uiucNeg-81-31-";
	for(short int i=0; i<2442; i++)
	{
		sstream.str("");
		sstream<<path<<prefix<<setfill('0')<<setw(2)<<i<<".pgm";

		Data d;
		d.image = LoadImage(sstream.str());
		d.label = -1.0/2442.0;
		trainingData.push_back(d);
	}

	return trainingData;
}

void Visualize(Mat matrix, string window)
{
	vector<int> compressionParams;
	compressionParams.push_back(cv::IMWRITE_JPEG_QUALITY);
	compressionParams.push_back(95);
	int rho = 9;

	if(window.compare("Visualize projector matrix")==0){
		normalize(matrix,matrix,0,255, NORM_MINMAX, CV_8UC1);
		string path = "images/Visualizing_tensor_projection_";
		imwrite(format("%s%d.jpg", (path).c_str(), rho), matrix, compressionParams);
		//imshow(window, matrix);
	}
	normalize(matrix,matrix,0,255, NORM_MINMAX, CV_8UC1);
	imshow(window, matrix);
	waitKey(0);
}

// Helper function to detect if input is a video file
bool isVideoFile(const string& path) {
	string ext_lower = path;
	transform(ext_lower.begin(), ext_lower.end(), ext_lower.begin(), ::tolower);
	vector<string> video_exts = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v", ".webm"};
	for (const auto& ext : video_exts) {
		if (ext_lower.length() >= ext.length() &&
		    ext_lower.substr(ext_lower.length() - ext.length()) == ext) {
			return true;
		}
	}
	return false;
}

// Process a video file frame-by-frame using LDA template
void processVideoWithTemplate(const string& videoPath, const Mat& templ, const string& outputPath) {
	VideoCapture cap;
	int frameCount;
	double fps;
	
	if (!openVideo(videoPath, cap, frameCount, fps)) {
		cerr << "Failed to open video: " << videoPath << endl;
		return;
	}

	vector<DetectionFrame> frames;
	Mat frame;
	int frameId = 0;

	cout << "Processing video frames..." << endl;

	while (cap.read(frame)) {
		if (frame.empty()) break;

		// process frame with LDA template matching
		vector<Rect> dets = processFrameWithTemplate(frame, templ, 0.7f);
		
		DetectionFrame df;
		df.frameId = frameId;
		df.image = frame.clone();
		df.detections = dets;
		frames.push_back(df);

		frameId++;
		if (frameId % 30 == 0) {
			cout << "Processed " << frameId << " frames..." << endl;
		}
	}

	cap.release();
	cout << "Total frames processed: " << frameId << endl;

	// write annotated video
	if (!frames.empty()) {
		int fourcc = VideoWriter::fourcc('m', 'p', '4', 'v');
		writeVideoWithDetections(outputPath, frames, fourcc, fps, 
		                          frames[0].image.cols, frames[0].image.rows);
	}
}

int main(int argc, char** argv) {


	if(argc < 3){
		cout<<"Usage ./LDA Separable <train-images-dir> <test-images-dir> <path-results> <rho={1, 3, 9} ";
		exit(-1);
	}

	else{
		cout<<"Path that contains train images: "<<argv[1]<<endl; //argv[1]trainFilesPath
		cout<<"Path that contains test images: "<<argv[2]<<endl;
		cout<<"Path to store the results: "<<argv[3]<<endl;
		cout<<"Select K- term projection tensor: 1, 3, or 9: "<<argv[4]<<endl;
		cout<<"Path for saving the visualization of projection tensor : "<<argv[5]<<endl;
	}


/*
	file_source f{"file.txt"};

	if (f.is_open()){
		stream<file_source> is{f};
		cout<<is.rdbuf()<<"\n";
		f.close();
	}
*/

	string trainFilesPath = argv[1];//+"uiucPos-81-31-";//"images/uiucTrain/"

	Mat projectorMat;

	int rho = stoi(argv[4]);

	vector<Data> trainingData = loadTrainingData(trainFilesPath);

	Mat mean = getMean(trainingData);

	Visualize(mean, "Mean");

	centerData(trainingData, mean);

	compute(trainingData, rho);

	cout<<"Computed 9-term projection";

	Mat temp = computeTemplate(rho);// sum of outer products

	Visualize(temp, "final");

	resize(temp, projectorMat,Size(81*3, 31*3));

	Visualize(projectorMat, "Visualize projector matrix");

	cout<<endl;
	cout<<"Tensor projection is of type "<<temp.type()<<endl;
	cout<<"Tensor projection is of size "<<temp.size()<<endl;


	ofstream handle("file.txt", ios::trunc);

	handle << projectorMat;

	string prefix = argv[2];
	string testFilesPath = prefix+"TEST_";//"images/uiucTest/";

//	string prefix = argv[3];//"images/uiucTestResults/TEST_RHO_"+to_string(rho)+"_";

	prefix = argv[3];
	string testFileResultsPath = prefix + "TEST_RHO_"+to_string(rho)+"_"; //"images/uiucTestResults/"

	string visualizeW = argv[5];

	// Check if test input is a video file
	if (isVideoFile(argv[2])) {
		cout << "Detected video input: " << argv[2] << endl;
		string videoOutputPath = string(argv[3]) + "output_video.mp4";
		processVideoWithTemplate(argv[2], temp, videoOutputPath);
	} else {
		// Process image sequence as before
		computeOnTestImages(temp, testFilesPath, testFileResultsPath, visualizeW, rho); //for rho = 9
	}

	waitKey(0);

	return 0;
}
