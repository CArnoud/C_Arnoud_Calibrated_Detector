#ifndef DETECTOR_H
#define DETECTOR_H

#include "Options.h"
#include "Info.h"
#include "BoundingBox.h"
#include "utils.h"
#include <fstream>
#include <string>

enum candidateType { SPARSE, FULL };

struct Config 
{
	float resizeImage;
	std::string detectorFileName;
	std::string groundTruthFileName;
	std::string dataSetDirectory;
	int firstFrame, lastFrame;

	bool displayDetections, saveFrames, saveDetectionsInText, useCalibration, showScore, useGroundTruth; 
	std::string outputFolder;
	int candidateGeneration;

	cv::Mat_<float> *projectionMatrix;
	cv::Mat_<float> *homographyMatrix;

	float supressionThreshold;

	float maxPedestrianWorldHeight;
	float minPedestrianWorldHeight; 

	Config(std::string config_file);
};


class Detector
{
public:
	Options opts; //opts contains the Pyramid

	//Clf clf;
	cv::Mat fids;
	cv::Mat thrs;
	cv::Mat child;
	cv::Mat hs;
	cv::Mat weights;
	cv::Mat depth;
	cv::Mat errs;
	cv::Mat losses;
	int treeDepth;
	Config config;

	//BB_Array_Array detections;

	double timeSpentInDetection;

	void exportDetectorModel(cv::String);
	void importDetectorModel(cv::String);
	void acfDetect(std::vector<std::string> imageNames, std::string dataSetDirectoryName, int firstFrame, int lastFrame);
	BB_Array nonMaximalSuppression(BB_Array bbs);

	Detector(Config _config): config(_config) { };

private:
	void showDetections(cv::Mat I, BB_Array detections, cv::String windowName, bool showScore);

	BoundingBox pyramidRowColumn2BoundingBox(int r, int c,  int modelHt, int modelWd, int ith_scale, int stride);

	void removeCandidateRegions(BB_Array detections, BB_Array& denseCandidates);

	//BB_Array* addCandidateRegions(BB_Array* candidates, int imageHeight, int imageWidth, int modelHeight, int modelWidth, float minPedestrianHeight,
	//									float maxPedestrianHeight, int shrink, cv::Mat_<float> &P, cv::Mat_<float> &H);

	BB_Array addCandidateRegions(BB_Array candidates, int imageHeight, int imageWidth, int modelHeight, int modelWidth, float minPedestrianHeight,
										float maxPedestrianHeight, int shrink, cv::Mat_<float> &P, cv::Mat_<float> &H);

	//BB_Array* generateCandidateRegions(BB_Array* candidates, int imageHeight, int imageWidth, int shrink, int modelHeight, int modelWidth, float minPedestrianHeight, 
	//										float maxPedestrianHeight, cv::Mat_<float> &P, cv::Mat_<float> &H);

	BB_Array generateCandidateRegions(BB_Array candidates, int imageHeight, int imageWidth, int shrink, int modelHeight, int modelWidth, 
											float minPedestrianHeight, float maxPedestrianHeight, cv::Mat_<float> &P, cv::Mat_<float> &H);

	BB_Array generateSparseCandidates(int modelWidth, int modelHeight, float minPedestrianHeight, float maxPedestrianHeight, int imageWidth, 
											int imageHeight, int shrink, cv::Mat_<float> &P, cv::Mat_<float> &H);

	BB_Array generateCandidates(int imageHeight, int imageWidth, int shrink, cv::Mat_<float> &P, cv::Mat_<float> &H, float BBwidth2heightRatio, 
									float meanHeight = 1800, float stdHeight = 100, float factorStdHeight = 2.0);

	int findClosestScaleFromBbox(int bbHeight, int imageHeight);

	int findClosestScaleFromBbox2(std::vector<Info> &pyramid, BoundingBox &bb,
												int modelHeight, double shrink);
	/*
	BB_Array applyCalibratedDetectorToFrame(BB_Array& bbox_candidates, std::vector<float*>& scales_chns, int *imageHeigths, int *imageWidths, int shrink, int modelHt, int modelWd, int stride, 
											float cascThr, float *thrs, float *hs, std::vector<uint32*>& scales_cids, uint32 *fids, uint32 *child, int nTreeNodes, 
											int nTrees, int treeDepth, int nChns, int imageWidth, int imageHeight, cv::Mat_<float> &P);*/
	BB_Array applyCalibratedDetectorToFrame(std::vector<Info>& pyramid, BB_Array& bbox_candidates, int shrink, 
											int modelHt, int modelWd, int stride, float cascThr, float *thrs, float *hs, std::vector<uint32*>& scales_cids, 
											uint32 *fids, uint32 *child, int nTreeNodes, int nTrees, int treeDepth, int nChns, int imageWidth, int imageHeight, 
											cv::Mat_<float> &P);

	BB_Array applyDetectorToFrame(std::vector<Info>& pyramid, int shrink, int modelHt, int modelWd, int stride, float cascThr, float *thrs, 
									float *hs, uint32 *fids, uint32 *child, int nTreeNodes, int nTrees, int treeDepth, int nChns);

	void bbTopLeft2PyramidRowColumn(int *r, int *c, BoundingBox& bb, int modelHt, int modelWd, int ith_scale, int stride);

	BB_Array ratedSuppression(BB_Array detections);

	BB_Array nonMaximalSuppressionSmart(BB_Array bbs, double meanHeight, double stdHeight);

	BB_Array candidateRegions;
};

#endif
