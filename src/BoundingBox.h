#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H

#include <cv.h>

class BoundingBox
{
public:
	BoundingBox();
	BoundingBox(int topLeftPointX, int topLeftPointY, int w, int h);
	BoundingBox(int topLeftPointX, int topLeftPointY, int w, int h, int s, float wHeight);
	//cv::Point2i  get_bottomRightPoint();
	void plot (cv::Mat &frame, cv::Scalar color, bool showScore);
	std::string toString (int frameIndex);
	void resize (float resizeFactor);

	// this was added to be able to sort BB_Array objects
	bool operator< (const BoundingBox &other) const 
	{
		//return score < other.score;
  		return score > other.score;
 	}

	cv::Point topLeftPoint;
	int width;
	int height;
	int scale;
	float worldHeight;
	float score;
};

typedef std::vector<BoundingBox> BB_Array;
typedef std::vector<BB_Array> BB_Array_Array;

#endif