#include "BoundingBox.h"

// empty constructor
BoundingBox::BoundingBox()
{
	topLeftPoint.x = -1;
	topLeftPoint.y = -1;
	width = 0;
	height = 0;
	scale = -1;
	worldHeight = 0.0;
	score = -1.0;
}

BoundingBox::BoundingBox(int topLeftPointX, int topLeftPointY, int w, int h)
{
	topLeftPoint.x = topLeftPointX;
	topLeftPoint.y = topLeftPointY;
	width = w;
	height = h;
	scale = -1;
	worldHeight = 0.0;
	score = -1.0;
}

BoundingBox::BoundingBox(int topLeftPointX, int topLeftPointY, int w, int h, int s, float wHeight)
{
	topLeftPoint.x = topLeftPointX;
	topLeftPoint.y = topLeftPointY;
	width = w;
	height = h;
	scale = s;
	worldHeight = wHeight;
	score = 0;
}

void BoundingBox::plot (cv::Mat &frame, cv::Scalar color, bool showScore)
{
	cv::Point br;
	br.x = topLeftPoint.x + width;
	br.y = topLeftPoint.y + height;
	cv::rectangle(frame, topLeftPoint, br, color, 2.0);

	if(showScore)
	{
		std::ostringstream scoreString;
		scoreString << round(score);
		putText(frame, scoreString.str(), cvPoint(topLeftPoint.x,topLeftPoint.y-2), cv::FONT_HERSHEY_PLAIN, 1.5, cvScalar(255,255,255));
	}
}

void BoundingBox::resize (float resizeFactor)
{
	topLeftPoint.x = topLeftPoint.x * resizeFactor;
	topLeftPoint.y = topLeftPoint.y * resizeFactor;
	height = height * resizeFactor;
	width = width * resizeFactor;
}

std::string BoundingBox::toString (int frameIndex)
{
	std::ostringstream result;

	result << "Frame: " << std::setw(4) << frameIndex << ", TopLeftPoint: (" << std::setw(4) << topLeftPoint.x << "," << std::setw(4) << topLeftPoint.y << 
			"), Height: " << std::setw(4) << height << ", Width: " << std::setw(3) << width << ", Score: " << score << std::endl;

	return result.str();
}

/*
cv::Point2i  BoundingBox::get_bottomRightPoint() {

	cv::Point2i br;
	br.x = topLeftPoint.x + width;
	br.y = topLeftPoint.y + height;

	return br;

}
*/