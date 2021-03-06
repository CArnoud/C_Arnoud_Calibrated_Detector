#include "Detector.h"

double gaussianFunction(double mean, double std, double x) {
	return exp(-pow(x-mean, 2)/(2*pow(std,2)));
}

Config::Config(std::string config_file) :
	resizeImage(1.0),
	firstFrame(0),
	lastFrame(999999),
	displayDetections(false),
	saveFrames(false),
	useCalibration(false),
	saveDetectionsInText(false),
	showScore(false),
	useGroundTruth(false),
	supressionThreshold(0.0),
	maxPedestrianWorldHeight(2000.0),
	minPedestrianWorldHeight(1400.0)
{
	std::ifstream in_file;
	in_file.open(config_file.c_str());

	if (in_file.is_open())
	{
		std::string token;
		while (in_file >> token) {
			if (token == "resizeImage") in_file >> resizeImage;
			else if (token == "firstFrame") in_file >> firstFrame;
			else if (token == "lastFrame") in_file >> lastFrame;
			else if (token == "detectorFileName") in_file >> detectorFileName;
			else if (token == "groundTruthFileName") in_file >> groundTruthFileName;
			else if (token == "dataSetDirectory") in_file >> dataSetDirectory;
			else if (token == "candidateGeneration") 
			{
				std::string candidateGenerationString;
				in_file >> candidateGenerationString;
				if (candidateGenerationString == "full")
					candidateGeneration = FULL;
				else
					candidateGeneration = SPARSE;
			}
			else if (token == "displayDetections") {
				std::string sbool;
				in_file >> sbool;
				displayDetections = (sbool == "true");
			}
			else if (token == "saveFrames") {
				std::string sbool;
				in_file >> sbool;
				saveFrames = (sbool == "true");
			}
			else if (token == "saveDetectionsInText") {
				std::string sbool;
				in_file >> sbool;
				saveDetectionsInText = (sbool == "true");
			}
			else if (token == "showScore") {
				std::string sbool;
				in_file >> sbool;
				showScore = (sbool == "true");
			}
			else if (token == "useGroundTruth") {
				std::string sbool;
				in_file >> sbool;
				useGroundTruth = (sbool == "true");
			}
			else if (token == "outputFolder") in_file >> outputFolder;
			else if (token == "supressionThreshold") in_file >> supressionThreshold;
			else if (token == "maxPedestrianWorldHeight") in_file >> maxPedestrianWorldHeight;
			else if (token == "minPedestrianWorldHeight") in_file >> minPedestrianWorldHeight;
			else if (token == "useCalibration")  {
				std::string sbool;
				in_file >> sbool;
				useCalibration = (sbool == "true");
			}
			else if (token == "projectionMatrix") {
				float *dP = new float[12];
				for (int i=0; i<12; ++i) {
					in_file >> dP[i];
				}

  				projectionMatrix = new cv::Mat_<float>(3, 4);
  				memcpy(projectionMatrix->data, dP, 3*4*sizeof(float));
				delete[] dP;

				if (resizeImage != 1.0) 
				{
					float s = resizeImage;
					float scale_matrix[9] = {s, 0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, 1.0};

					cv::Mat_<float> S(3, 3, scale_matrix);
					*(projectionMatrix) = S * *(projectionMatrix);
				}

				homographyMatrix = new cv::Mat_<float>(3,3);

				homographyMatrix->at<float>(0,0) = projectionMatrix->at<float>(0,0);
				homographyMatrix->at<float>(0,1) = projectionMatrix->at<float>(0,1);
				homographyMatrix->at<float>(0,2) = projectionMatrix->at<float>(0,3);
				homographyMatrix->at<float>(1,0) = projectionMatrix->at<float>(1,0);
				homographyMatrix->at<float>(1,1) = projectionMatrix->at<float>(1,1);
				homographyMatrix->at<float>(1,2) = projectionMatrix->at<float>(1,3);
				homographyMatrix->at<float>(2,0) = projectionMatrix->at<float>(2,0);
				homographyMatrix->at<float>(2,1) = projectionMatrix->at<float>(2,1);
				homographyMatrix->at<float>(2,2) = projectionMatrix->at<float>(2,3);
			}
			else {
				std::cout << "Token not recognized!" << std::endl;
			}
		}

		in_file.close();
	}
	else
		std::cout << " # Configuration file named " << config_file << " was not found.\n"; 
}

// i dont know if its gonna be needed but this is start
void Detector::exportDetectorModel(cv::String fileName)
{
	cv::FileStorage xml;
	
	xml.open(fileName, cv::FileStorage::WRITE);

	xml << "opts" << "{";
		xml << "pPyramid" << "{";
			xml << "pChns" << "{";
				xml << "shrink" << opts.pPyramid.pChns.shrink;
				xml << "pColor" << "{";
					xml << "enabled" << opts.pPyramid.pChns.pColor.enabled;
				xml << "}";
			xml << "}";
		xml << "}";
	xml << "stride" << opts.stride;
	xml << "}";
	
	//xml << "clf" << this->clf;

	xml.release();
}

// reads the detector model from the xml model
void Detector::importDetectorModel(cv::String fileName)
{
	cv::FileStorage xml;

	xml.open(fileName, cv::FileStorage::READ);

	if (!xml.isOpened())
	{
		std::cerr << " # Failed to open " << fileName << std::endl;
	}
	else
	{
		opts.readOptions(xml["detector"]["opts"]);
	
		xml["detector"]["clf"]["fids"] >> fids;
		xml["detector"]["clf"]["child"] >> child;
		xml["detector"]["clf"]["thrs"] >> thrs;
		xml["detector"]["clf"]["hs"] >> hs;
		xml["detector"]["clf"]["weights"] >> weights;
		xml["detector"]["clf"]["depth"] >> depth;
		xml["detector"]["clf"]["errs"] >> errs;
		xml["detector"]["clf"]["losses"] >> losses;		
		xml["detector"]["clf"]["treeDepth"] >> treeDepth;	

		timeSpentInDetection = 0;

		xml.release();
	}
}

void Detector::showDetections(cv::Mat I, BB_Array detections, cv::String windowName, bool showScore)
{
	cv::Mat img = I.clone();
	for (int j = 0; j<detections.size(); j++) 
		//detections[j].plot(img, cv::Scalar(0,255,0), showScore);
		detections[j].plot(img, cv::Scalar(255,0,0), showScore);

	cv::imshow(windowName, img);
}

BB_Array_Array readGroundTruth(std::string fileName, float resize)
{
	BB_Array_Array groundTruth;
	std::ifstream in_file;
	in_file.open(fileName.c_str());

	if (in_file.is_open())
	{
		std::string token;
		while (in_file >> token && token != "</dataset>")
		{
			BB_Array frameBBs;
			while (in_file >> token && token != "</frame>") {
				if (token == "<box>") 
				{
					float height, width, xCenter, yCenter;
					while(token != "</box>")
					{
						in_file >> token;
						if (token == "<height>")
							in_file >> height;						
						if (token == "<width>")
							in_file >> width;
						if (token == "<xCenter>")
							in_file >> xCenter;
						if (token == "<yCenter>")
							in_file >> yCenter;
					}
					BoundingBox newBB(xCenter-width/2, yCenter-height/2, width, height);
					newBB.resize(resize);
					frameBBs.push_back(newBB);
				}
			}

			groundTruth.push_back(frameBBs);
		}
	}
	else
		std::cout << " # Ground Truth file named " << fileName << " was not found!\n";

	return groundTruth;
}

BB_Array_Array readGroundTruthTopFile(std::string fileName, float resize)
{ //frameNumber, headValid, bodyValid, headLeft, headTop, headRight, headBottom, bodyLeft, bodyTop, bodyRight, bodyBottom
	BB_Array_Array groundTruth;
	std::ifstream infile;
	infile.open(fileName.c_str());

	if (infile.is_open())
	{
		int frame, returnPlace = 0;
		while(infile >> frame)
		{
			infile.seekg(returnPlace);
			infile >> frame;
			int newFrame = frame;
			BB_Array frameBBs;

			while (newFrame == frame)
			{

				std::string notUsed;
				float topLeftX, topLeftY, bottomRightX, bottomRightY;
				infile >> notUsed; 		// headValid
				infile >> notUsed; 		// bodyValid
				infile >> notUsed; 		// headLeft
				infile >> topLeftY; 	// headTop
				infile >> notUsed; 		// headRight
				infile >> notUsed; 		// headBottom
				infile >> topLeftX;		// bodyLeft
				infile >> notUsed; 		// bodyTop
				infile >> bottomRightX;	// bodyRight
				infile >> bottomRightY;	// bodyBottom

				BoundingBox newBB(topLeftX, topLeftY, bottomRightX-topLeftX, bottomRightY-topLeftY);
				newBB.resize(resize);
				frameBBs.push_back(newBB);

				returnPlace = infile.tellg();
				if (!(infile >> newFrame))
					break;
			}

			groundTruth.push_back(frameBBs);
		}
	}
	else
		std::cout << " # Ground Truth file named " << fileName << " was not found!\n";

	return groundTruth;
}

void printDetections(BB_Array detections, int frameIndex)
{
	std::cout << "Detections in frame " << frameIndex << ":\n";
	for (int i=0; i < detections.size(); i++)
		std::cout << detections[i].toString(frameIndex) << std::endl;
	std::cout << std::endl;
}

bool isBBinsideRegion(BoundingBox region, int bottomLeftX, int bottomLeftY)
{
	if (bottomLeftX >= region.topLeftPoint.x && 
		bottomLeftX <= region.topLeftPoint.x + region.width &&
		bottomLeftY <= region.topLeftPoint.y + region.height &&
		bottomLeftY >= region.topLeftPoint.y)
		return true;
	else
		return false;
}

void Detector::removeCandidateRegions(BB_Array detections, BB_Array& denseCandidates)
{
	for (int i=0; i < candidateRegions.size(); i++)
	{
		int j=0;
		bool found = false;
		while (!found && j < detections.size())
		{
			found=isBBinsideRegion(candidateRegions[i], detections[j].topLeftPoint.x, detections[j].topLeftPoint.y+detections[j].height);

			j++;
		}

		if (!found)
		{
			// remove the region from the region vector
			std::swap(candidateRegions[i], candidateRegions.back());
			candidateRegions.pop_back();
			i--;
		}
	}

	for (int i=0; i < denseCandidates.size(); i++)
	{
		int j=0;
		bool insideDenseRegion=false;

		while(!insideDenseRegion && j < candidateRegions.size())
		{
			insideDenseRegion=isBBinsideRegion(candidateRegions[j], denseCandidates[i].topLeftPoint.x, denseCandidates[i].topLeftPoint.y+denseCandidates[i].height);
			j++;
		}

		// delete the candidate bounding box if it is not inside on of the dense regions that are stiull necessary
		if(!insideDenseRegion)
		{
			std::swap(denseCandidates[i], denseCandidates.back());
			denseCandidates.pop_back();
			i--;
		}
	}
}

BB_Array Detector::addCandidateRegions(BB_Array candidates, int imageHeight, int imageWidth, int modelHeight, int modelWidth, float minPedestrianHeight,
										float maxPedestrianHeight, int shrink, cv::Mat_<float> &P, cv::Mat_<float> &H)
{
	BB_Array result;
	BB_Array sparseDetections;

	for (int i = 0; i < candidates.size(); i++)
	{
		bool covered = false;
		int j=0;
		while (!covered && j < candidateRegions.size())
		{
			covered = isBBinsideRegion(candidateRegions[j], candidates[i].topLeftPoint.x, candidates[i].topLeftPoint.y+candidates[i].height); 

			j++;
		}

		if (!covered)
		{
			// test if there isn't a new region already for this one
			int l=0;
			bool discarded = false;

			while (!discarded && l < sparseDetections.size())
			{
				if (candidates[i].topLeftPoint.x >= sparseDetections[l].topLeftPoint.x-sparseDetections[l].width &&
				candidates[i].topLeftPoint.x <= sparseDetections[l].topLeftPoint.x+sparseDetections[l].width &&
				candidates[i].topLeftPoint.y >= sparseDetections[l].topLeftPoint.y-(sparseDetections[l].height/2) &&
				candidates[i].topLeftPoint.y <= sparseDetections[l].topLeftPoint.y+(sparseDetections[l].height/2)) 
				{
					discarded = true;

					if (candidates[i].score > sparseDetections[l].score)
					{
						sparseDetections[l].topLeftPoint.x = candidates[i].topLeftPoint.x;
						sparseDetections[l].topLeftPoint.y = candidates[i].topLeftPoint.y;
					}

					if (candidates[i].height > sparseDetections[l].height)
						sparseDetections[l].height = candidates[i].height;

					if (candidates[i].width > sparseDetections[l].width)
						sparseDetections[l].width = candidates[i].width;
				}

				l++;
			}

			if (!discarded)
			{
				sparseDetections.push_back(candidates[i]);
			}
		}
	}

	for (int i=0; i < sparseDetections.size(); i++)
	{
		int j=0;
		bool covered = false;

		while (!covered && j < sparseDetections.size())
		{
			covered=isBBinsideRegion(candidateRegions[j], sparseDetections[i].topLeftPoint.x, sparseDetections[i].topLeftPoint.y+sparseDetections[i].height); 		

			j++;
		}

		if (!covered)
		{
			// add the candidates in the new region to newCandidates and add the region 
			int v = sparseDetections[i].topLeftPoint.y + (sparseDetections[i].height/2);
			int maxV = sparseDetections[i].topLeftPoint.y + (3*sparseDetections[i].height/2);
			if (maxV > imageHeight)
				maxV = imageHeight;

			int maxU = sparseDetections[i].topLeftPoint.x + sparseDetections[i].width;
			if (maxU > imageWidth-modelWidth)
				maxU = imageWidth-modelWidth;

			int regionU = sparseDetections[i].topLeftPoint.x - sparseDetections[i].width;
			if (regionU < 0)
				regionU = 0;
			BoundingBox region(regionU, v, maxU-regionU, maxV-v);
			candidateRegions.push_back(region);

			while (v < maxV)
			{
				int u = sparseDetections[i].topLeftPoint.x - sparseDetections[i].width;
				if (u < 0)
					u = 0;
				
				while (u < maxU)
				{
					// we start at the top of the region
					int head_v = sparseDetections[i].topLeftPoint.y - (sparseDetections[i].height/2); 
					if (head_v < 0)
						head_v = 0;
					
					double bbWorldHeight = findWorldHeight(u, v, head_v, P, H);
					while (head_v < v-modelHeight && bbWorldHeight > maxPedestrianHeight)
					{
						head_v = head_v + shrink;
						bbWorldHeight = findWorldHeight(u, v, head_v, P, H);
					}

					if (bbWorldHeight <= maxPedestrianHeight)
					{
						// found the biggest valid bounding box in the point
						int bbHeight = v-head_v;
						int bbWidth = bbHeight*modelWidth/modelHeight;
						int bbScale; 

						if (u + bbWidth <= imageWidth)
						{
							bbScale = findClosestScaleFromBbox(bbHeight, imageHeight);
							BoundingBox maxCandidate(u, head_v, bbWidth, bbHeight, bbScale, bbWorldHeight);
							result.push_back(maxCandidate);
						}

						int previousScale = -1;

						// now, find other valid bounding boxes
						while (bbWorldHeight > minPedestrianHeight && head_v < v-modelHeight)
						{
							head_v = head_v + shrink;
							bbWorldHeight = findWorldHeight(u, v, head_v, P, H);

							if (bbWorldHeight >= minPedestrianHeight)
							{
								bbHeight = v-head_v;
								bbScale = findClosestScaleFromBbox(bbHeight, imageHeight);

								if (bbScale != previousScale)
								{
									previousScale = bbScale;

									bbWidth = bbHeight*modelWidth/modelHeight;

									if (u + bbWidth <= imageWidth)
									{
										BoundingBox newCandidate(u, head_v, bbWidth, bbHeight, bbScale, bbWorldHeight);
										result.push_back(newCandidate);
									}
								}
							}
						}
					}
					u = u + shrink;
				}
				v = v + shrink;
			}
		}
	}

	sparseDetections.clear();

	return result;
}

BB_Array Detector::generateCandidateRegions(BB_Array candidates, int imageHeight, int imageWidth, int shrink, int modelHeight, int modelWidth, 
											float minPedestrianHeight, float maxPedestrianHeight, cv::Mat_<float> &P, cv::Mat_<float> &H)
{
	BB_Array result;
	BB_Array tempResult;

	// sort the candidates by score
	BB_Array sortedCandidates(candidates.size());
	for (int i=0; i < candidates.size(); i++)
	{
		sortedCandidates[i] = candidates[i];
	} 
	std::sort(sortedCandidates.begin(), sortedCandidates.begin()+sortedCandidates.size());

	// keeps only the biggest bounding box of each region (could be relocated to the classifier)
	for (int i = 0; i < candidates.size(); ++i)
	{
		bool discard = false;
		int j=0;
		while (j < tempResult.size() && !discard)
		{
			int regionTop = tempResult[j].topLeftPoint.y-(tempResult[j].height/2);
			if (regionTop < 0)
				regionTop = 0;

			int regionBottom = tempResult[j].topLeftPoint.y+(tempResult[j].height/2);
			if (regionBottom > imageHeight)
				regionBottom = imageHeight;

			int regionLeft = tempResult[j].topLeftPoint.x-tempResult[j].width;
			if (regionLeft < 0)
				regionLeft = 0;

			int regionRight = tempResult[j].topLeftPoint.x+tempResult[j].width;
			if (regionRight > imageWidth)
				regionRight = imageWidth;

			// revisar esse cálculo!
			if (sortedCandidates[i].topLeftPoint.x >= regionLeft &&
				sortedCandidates[i].topLeftPoint.x <= regionRight &&
				sortedCandidates[i].topLeftPoint.y >= regionTop &&
				sortedCandidates[i].topLeftPoint.y <= regionBottom) 
			{
				discard = true;

				if (sortedCandidates[i].height > tempResult[j].height)
				{
					tempResult[j].height = sortedCandidates[i].height;
				}

				if (sortedCandidates[i].width > tempResult[j].width)
				{
					tempResult[j].width = sortedCandidates[i].width;
				}
			}

			j++;
		}

		if (!discard)
			tempResult.push_back(sortedCandidates[i]);
	}

	/*
	// std::cout << tempResult.size() << " regions to be created\n";
	cv::Mat image = cv::imread("/home/c_arnoud/datasets/View_001/frame_0730.jpg");
	cv::resize(image, image, cv::Size(), 1.5, 1.5);
	showDetections(image, tempResult, "region source");
	*/

	// here, we need to create the regions around the remaining bounding boxes
	for (int i=0; i < tempResult.size(); i++)
	{
		int v = tempResult[i].topLeftPoint.y + (tempResult[i].height/2);
		int maxV = tempResult[i].topLeftPoint.y + (3*tempResult[i].height/2);
		if (maxV > imageHeight)
			maxV = imageHeight;

		int maxU = tempResult[i].topLeftPoint.x + tempResult[i].width;
		if (maxU > imageWidth-modelWidth)
			maxU = imageWidth-modelWidth;

		int regionU = tempResult[i].topLeftPoint.x - tempResult[i].width;
		if (regionU < 0)
			regionU = 0;
		// correct version
		BoundingBox region(regionU, v, maxU-regionU, maxV-v);
		// test version
		//BoundingBox region(regionU, tempResult[i].topLeftPoint.y-(tempResult[i].height/2), maxU-regionU+tempResult[i].width, maxV-(tempResult[i].topLeftPoint.y-(tempResult[i].height/2)));
		candidateRegions.push_back(region);

		while (v < maxV)
		{
			int u = tempResult[i].topLeftPoint.x - tempResult[i].width;
			if (u < 0)
				u = 0;
			
			while (u < maxU)
			{
				// we start at the top of the region
				int head_v = tempResult[i].topLeftPoint.y - (tempResult[i].height/2); 
				if (head_v < 0)
					head_v = 0;
				
				double bbWorldHeight = findWorldHeight(u, v, head_v, P, H);
				while (head_v < v-modelHeight && bbWorldHeight > maxPedestrianHeight)
				{
					head_v = head_v + shrink;
					bbWorldHeight = findWorldHeight(u, v, head_v, P, H);
				}

				if (bbWorldHeight <= maxPedestrianHeight)
				{
					// found the biggest valid bounding box in the point
					int bbHeight = v-head_v;
					int bbWidth = bbHeight*modelWidth/modelHeight;
					int bbScale; 

					if (u + bbWidth <= imageWidth)
					{
						bbScale = findClosestScaleFromBbox(bbHeight, imageHeight);
						BoundingBox maxCandidate(u, head_v, bbWidth, bbHeight, bbScale, bbWorldHeight);
						result.push_back(maxCandidate);
					}

					int previousScale = -1;

					// now, find other valid bounding boxes
					while (bbWorldHeight > minPedestrianHeight && head_v < v-modelHeight)
					{
						head_v = head_v + shrink;
						bbWorldHeight = findWorldHeight(u, v, head_v, P, H);

						if (bbWorldHeight >= minPedestrianHeight)
						{
							bbHeight = v-head_v;
							bbScale = findClosestScaleFromBbox(bbHeight, imageHeight);

							if (bbScale != previousScale)
							{
								previousScale = bbScale;

								bbWidth = bbHeight*modelWidth/modelHeight;

								if (u + bbWidth <= imageWidth)
								{
									BoundingBox newCandidate(u, head_v, bbWidth, bbHeight, bbScale, bbWorldHeight);
									result.push_back(newCandidate);
								}
							}
						}
					}
				}
				u = u + shrink;
			}
			v = v + shrink;
		}

		/*
		std::cout << result->size() << " candidates created so far.\n";
		std::cin.get();
		*/
	}

	return result;
} 

BB_Array Detector::generateSparseCandidates(int modelWidth, int modelHeight, float minPedestrianHeight, float maxPedestrianHeight, int imageWidth, 
											int imageHeight, int shrink, cv::Mat_<float> &P, cv::Mat_<float> &H) 
{
	int u;
	int v = modelHeight;
	BB_Array candidates;
	int iterationCount=0;

	while (v < imageHeight)
	{
		int sumBBHeights = 0;
		u=0 + ((iterationCount%2)*modelWidth)/2;

		while (u < imageWidth-modelWidth)
		{
			int head_v = 0;
			double bbWorldHeight = findWorldHeight(u, v, head_v, P, H);

			while (head_v < v-modelHeight && bbWorldHeight > maxPedestrianHeight)
			{
				head_v = head_v + shrink;
				bbWorldHeight = findWorldHeight(u, v, head_v, P, H);
			}

			if (bbWorldHeight <= maxPedestrianHeight)
			{
				// found the biggest valid bounding box in the point
				int bbHeight = v-head_v;
				int bbWidth = bbHeight*modelWidth/modelHeight;
				int bbScale;

				if (u + bbWidth <= imageWidth)
				{
					bbScale = findClosestScaleFromBbox(bbHeight, imageHeight);
					BoundingBox maxCandidate(u, head_v, bbWidth, bbHeight, bbScale, bbWorldHeight);
					candidates.push_back(maxCandidate);
					sumBBHeights = sumBBHeights + bbHeight;
				}

				int previousScale=-1;

				// now, find other valid bounding boxes
				while (bbWorldHeight > minPedestrianHeight && head_v < v-modelHeight)
				{
					head_v = head_v + shrink;
					bbWorldHeight = findWorldHeight(u, v, head_v, P, H);

					if (bbWorldHeight >= minPedestrianHeight)
					{
						bbHeight = v-head_v;
						bbWidth = bbHeight*modelWidth/modelHeight;
						bbScale = findClosestScaleFromBbox(bbHeight, imageHeight);

						if (u + bbWidth <= imageWidth && bbScale != previousScale)
						{
							previousScale = bbScale;
							BoundingBox newCandidate(u, head_v, bbWidth, bbHeight, bbScale, bbWorldHeight);
							candidates.push_back(newCandidate);
							sumBBHeights = sumBBHeights + bbHeight;
						}
					}
				}
			}

			u = u + floor(modelWidth/2);
		}

		iterationCount++;
		v = v + floor(modelHeight/2);
	}

	return candidates;
}

BB_Array Detector::generateCandidates(int imageHeight, int imageWidth, int shrink, cv::Mat_<float> &P, cv::Mat_<float> &H, float BBwidth2heightRatio, 
										float meanHeight/* = 1.7m*/, float stdHeight/* = 0.1m*/, float factorStdHeight/* = 2.0*/) 
{

	// there is a set of parameters here that are hard coded, but should
	// be read from a file or something...
	cv::Mat_<float> P3 = P.col(2);

	float aspectRatio = BBwidth2heightRatio;
	float minImageHeight = 80;

	float stepHeight = 100;
	int totalCandidates = 0;

	BB_Array candidates;
	double max_h = 0;
	
	cv::Mat_<float> H_inv = H.inv();

	// create foot points using the pixels of the image
	for (int u = 0; u < imageWidth; u+=shrink) {
		for (int v = minImageHeight; v < imageHeight; v+=shrink ) {

			float Xw = (H_inv(0,0)*u + H_inv(0,1)*v + H_inv(0,2))/(H_inv(2,0)*u + H_inv(2,1)*v + H_inv(2,2));
			float Yw = (H_inv(1,0)*u + H_inv(1,1)*v + H_inv(1,2))/(H_inv(2,0)*u + H_inv(2,1)*v + H_inv(2,2));

			// now create candidates at different heights
			for (float h = -stdHeight * factorStdHeight; h <= stdHeight * factorStdHeight; h+= stepHeight) {
				float wHeight = meanHeight + h;

				int head_v = (int)((Xw*P(1,0) + Yw*P(1,1) + wHeight*P(1,2) + P(1,3))/(Xw*P(2,0) + Yw*P(2,1) + wHeight*P(2,2) + P(2,3)));
				int i_height = v - head_v;

				if (i_height >= minImageHeight) {
					int head_u = (int)((Xw*P(0,0) + Yw*P(0,1) + wHeight*P(0,2) + P(0,3))/(Xw*P(2,0) + Yw*P(2,1) + wHeight*P(2,2) + P(2,3)));

					BoundingBox bb;

					int i_width = i_height*aspectRatio;

				    bb.topLeftPoint.x = (int)(((u + head_u)/2.0) - (i_width/2.0));
				    bb.topLeftPoint.y = head_v;
				    bb.width          = i_width;
				    bb.height         = i_height;
				    bb.worldHeight   = wHeight;

				    if (bb.topLeftPoint.x >= 0 && bb.topLeftPoint.x+bb.width < imageWidth && 
						    bb.topLeftPoint.y >= 0 && bb.topLeftPoint.y+bb.height < imageHeight &&
						    bb.height >= minImageHeight) {
						if (bb.height > max_h) 
							max_h = bb.height;
						bb.scale = findClosestScaleFromBbox(bb.height, imageHeight);
						candidates.push_back(bb);
					}
				}

				totalCandidates++;

			}
		}
	}
 
	return candidates;
}

// This function returns the scale in which the pyramid will be better fitted
int Detector::findClosestScaleFromBbox(int bbHeight, int imageHeight)
{
	// actually here is the size of the the image that changes, the model stays the same
	// to see the best fit for the bounding box, one must find the relation between the original
	// and then find the closest to the size of the bounding box
	float min_dist = imageHeight;
	int i_min = -1;

	for (int i = 0; i < opts.pPyramid.computedScales; i++) 
	{
		float diff = fabs(opts.modelDs[0]/opts.pPyramid.scales[i] - bbHeight);

		if (diff < min_dist) {
			i_min = i;
			min_dist = diff;
		}
		else break;
	}

	return i_min;
}

void Detector::bbTopLeft2PyramidRowColumn(int *r, int *c, BoundingBox& bb, int modelHt, int modelWd, int ith_scale, int stride) 
{
	double s1, s2, sw, sh, tlx, tly;

	s1 = (modelHt-double(opts.modelDs[0]))/2-opts.pPyramid.pad[0];
	s2 = (modelWd-double(opts.modelDs[1]))/2-opts.pPyramid.pad[1];

	sw = opts.pPyramid.scales_w[ith_scale];
	sh = opts.pPyramid.scales_h[ith_scale];

	tlx = (double)bb.topLeftPoint.x;
	tly = (double)bb.topLeftPoint.y;

	double fc = (sw*tlx - s2)/(double)stride;
	double fr = (sh*tly - s1)/(double)stride;

	*r = (int)fr;
	*c = (int)fc;
}

BoundingBox Detector::pyramidRowColumn2BoundingBox(int r, int c,  int modelHt, int modelWd, int ith_scale, int stride) 
{

	double shift[2];
	shift[0] = (modelHt-double(opts.modelDs[0]))/2-opts.pPyramid.pad[0];
	shift[1] = (modelWd-double(opts.modelDs[1]))/2-opts.pPyramid.pad[1];

	BoundingBox bb;
	bb.topLeftPoint.x = c*stride;
	bb.topLeftPoint.x = (bb.topLeftPoint.x+shift[1])/opts.pPyramid.scales_w[ith_scale];
	bb.topLeftPoint.y = r*stride;
	bb.topLeftPoint.y = (bb.topLeftPoint.y+shift[0])/opts.pPyramid.scales_h[ith_scale];
	bb.height = opts.modelDs[0]/opts.pPyramid.scales[ith_scale];
	bb.width = opts.modelDs[1]/opts.pPyramid.scales[ith_scale];
	bb.scale = ith_scale;

	return bb;
}

double calculateOverlap(BoundingBox bb1, BoundingBox bb2)
{
	double overlap = 0.0;
	double xei, xej, xmin, xsMax, iw;
	double yei, yej, ymin, ysMax, ih;
	xei = bb1.topLeftPoint.x + bb1.width;
	xej = bb2.topLeftPoint.x + bb2.width;
	xmin = xej;			
	if (xei < xej)
		xmin = xei;
	xsMax = bb1.topLeftPoint.x;
	if (bb2.topLeftPoint.x > bb1.topLeftPoint.x)
		xsMax = bb2.topLeftPoint.x;
	iw = xmin - xsMax;
	yei = bb1.topLeftPoint.y + bb1.height;
	yej = bb2.topLeftPoint.y + bb2.height;
	ymin = yej;			
	if (yei < yej)
		ymin = yei;
	ysMax = bb1.topLeftPoint.y;
	if (bb2.topLeftPoint.y > bb1.topLeftPoint.y)
		ysMax = bb2.topLeftPoint.y;
	ih = ymin - ysMax;
	if (iw  > 0 && ih > 0)
	{
		overlap = iw * ih;
		double u = bb1.height*bb1.width + bb2.height*bb2.width-overlap;
		overlap = overlap/u;
	}
	return overlap;
}

int findFramesCoveredPedestrians(BB_Array groundTruth, BB_Array frameDetections)
{
	int coveredPedestrians = 0;
	
	for (int i=0; i < groundTruth.size(); i++)
	{
		bool found=false;
		int j = 0;
		while (!found && j < frameDetections.size())
		{
			if (calculateOverlap(groundTruth[i],frameDetections[j]) >= 0.5)
			{
				coveredPedestrians++;
				found=true;
			}
			j++;
		}
	}
	
	return coveredPedestrians;
}

// this procedure was just copied verbatim
inline void getChild(float *chns1, uint32 *cids, uint32 *fids, float *thrs, uint32 offset, uint32 &k0, uint32 &k)
{
  float ftr = chns1[cids[fids[k]]];
  k = (ftr<thrs[k]) ? 1 : 2;
  k0=k+=k0*2; k+=offset;
}

/*
BB_Array Detector::applyCalibratedDetectorToFrame(BB_Array& bbox_candidates, std::vector<float*>& scales_chns, int *imageHeigths, int *imageWidths, int shrink, 
											int modelHt, int modelWd, int stride, float cascThr, float *thrs, float *hs, std::vector<uint32*>& scales_cids, 
											uint32 *fids, uint32 *child, int nTreeNodes, int nTrees, int treeDepth, int nChns, int imageWidth, int imageHeight, 
											cv::Mat_<float> &P) */
BB_Array Detector::applyCalibratedDetectorToFrame(std::vector<Info>& pyramid, BB_Array& bbox_candidates, int shrink, int modelHt, int modelWd, int stride,
												float cascThr, float *thrs, float *hs, std::vector<uint32*>& scales_cids, uint32 *fids, uint32 *child,
											 	int nTreeNodes, int nTrees, int treeDepth, int nChns, int imageWidth, int imageHeight, cv::Mat_<float> &P)											
{
	BB_Array result;

	std::vector<float*> scales_chns(opts.pPyramid.computedScales, NULL);
	for (int j=0; j < opts.pPyramid.computedScales; j++) 
	{
		int height = pyramid[j].image.rows;
		int width = pyramid[j].image.cols;

		int channels = opts.pPyramid.pChns.pColor.nChannels + opts.pPyramid.pChns.pGradMag.nChannels + opts.pPyramid.pChns.pGradHist.nChannels;
		float* chns = (float*)malloc(height*width*channels*sizeof(float));
		features2floatArray(pyramid[j], chns, height, width, opts.pPyramid.pChns.pColor.nChannels, opts.pPyramid.pChns.pGradMag.nChannels, opts.pPyramid.pChns.pGradHist.nChannels);
		scales_chns[j] = chns;
	}

	for (int i = 0; i < bbox_candidates.size(); i++) 
	{
		int ith_scale = bbox_candidates[i].scale;
		
		int height = pyramid[ith_scale].image.rows;                                                             
		int width = pyramid[ith_scale].image.cols;  

		// r and c are defined by the candidate itself
		int r, c;
		bbTopLeft2PyramidRowColumn(&r, &c, bbox_candidates[i], modelHt, modelWd, ith_scale, stride);
		
		float h=0, *chns1=scales_chns[ith_scale]+(r*stride/shrink) + (c*stride/shrink)*height;
	    
	    if( treeDepth==1 ) {
	      // specialized case for treeDepth==1
	      for( int t = 0; t < nTrees; t++ ) {
	        uint32 offset=t*nTreeNodes, k=offset, k0=0;
	        getChild(chns1,scales_cids[ith_scale],fids,thrs,offset,k0,k);
	        h += hs[k]; if( h<=cascThr ) break;
	      }
	    } else if( treeDepth==2 ) {
	      // specialized case for treeDepth==2
	      for( int t = 0; t < nTrees; t++ ) {
	        uint32 offset=t*nTreeNodes, k=offset, k0=0;

	        getChild(chns1,scales_cids[ith_scale],fids,thrs,offset,k0,k);
	        getChild(chns1,scales_cids[ith_scale],fids,thrs,offset,k0,k);
	        
	        h += hs[k]; if( h<=cascThr ) break;
	      }
	    } else if( treeDepth>2) {
	      // specialized case for treeDepth>2
	      for( int t = 0; t < nTrees; t++ ) {
	        uint32 offset=t*nTreeNodes, k=offset, k0=0;
	        for( int i=0; i<treeDepth; i++ )
	          getChild(chns1,scales_cids[ith_scale],fids,thrs,offset,k0,k);
	        h += hs[k]; if( h<=cascThr ) break;
	      }
	    } else {
	      // general case (variable tree depth)
	      for( int t = 0; t < nTrees; t++ ) {
	        uint32 offset=t*nTreeNodes, k=offset, k0=k;
	        while( child[k] ) {
	          float ftr = chns1[scales_cids[ith_scale][fids[k]]];
	          k = (ftr<thrs[k]) ? 1 : 0;
	          k0 = k = child[k0]-k+offset;
	        }
	        h += hs[k]; if( h<=cascThr ) break;
	      }
	    }

	    //double hf = h*gaussianFunction(1.8, 0.1, bbox_candidates[i].worldHeight);
	    //if (hf>1.0)
	    if(h>cascThr)
	    {
			BoundingBox detection(bbox_candidates[i]);
			detection.score = h;
			detection.scale = ith_scale;
	    	result.push_back(detection);
	    }
	
	}

	//free the memory used to pre-allocate indexes
	for (int i=0; i < opts.pPyramid.computedScales; i++) {
		free(scales_chns[i]);
	}

	return result;
}

// copies the Dóllar detection
BB_Array Detector::applyDetectorToFrame(std::vector<Info>& pyramid, int shrink, int modelHt, int modelWd, int stride, float cascThr, float *thrs, float *hs, 
										uint32 *fids, uint32 *child, int nTreeNodes, int nTrees, int treeDepth, int nChns)
{
	BB_Array result;

	// this became a simple loop because we will apply just one detector here, 
	// to apply multiple detector models you need to create multiple Detector objects. 
	for (int i = 0; i < opts.pPyramid.computedScales; i++)
	{
		// in the original file: *chnsSize = mxGetDimensions(P.data{i});
		// const int height = (int) chnsSize[0];
  		// const int width = (int) chnsSize[1];
  		// const int nChns = mxGetNumberOfDimensions(prhs[0])<=2 ? 1 : (int) chnsSize[2];
		int height = pyramid[i].image.rows;
		int width = pyramid[i].image.cols;
		int channels = opts.pPyramid.pChns.pColor.nChannels + opts.pPyramid.pChns.pGradMag.nChannels + opts.pPyramid.pChns.pGradHist.nChannels;

		int height1 = (int)ceil(float(height*shrink-modelHt+1)/stride);
		int width1 = (int)ceil(float(width*shrink-modelWd+1)/stride);
		float* chns = (float*)malloc(height*width*channels*sizeof(float));
		features2floatArray(pyramid[i], chns, height, width,  opts.pPyramid.pChns.pColor.nChannels, opts.pPyramid.pChns.pGradMag.nChannels, opts.pPyramid.pChns.pGradHist.nChannels);
		
		// construct cids array
	  	int nFtrs = modelHt/shrink*modelWd/shrink*nChns;
	  	uint32 *cids = new uint32[nFtrs]; int m=0;
	  	for( int z=0; z<nChns; z++ ) {
	    	for( int c=0; c<modelWd/shrink; c++ ) {
	      		for( int r=0; r<modelHt/shrink; r++ ) {
	        		cids[m++] = z*width*height + c*height + r;
	        	}
	        }
	    }

		// apply classifier to each patch
  		std::vector<int> rs, cs; std::vector<float> hs1;
  		for( int c=0; c<width1; c++ ) 
  		{
  			for( int r=0; r<height1; r++ ) 
  			{
			    float h=0, *chns1=chns+(r*stride/shrink) + (c*stride/shrink)*height;
			    if( treeDepth==1 ) {
			      // specialized case for treeDepth==1
			      for( int t = 0; t < nTrees; t++ ) {
			        uint32 offset=t*nTreeNodes, k=offset, k0=0;
			        getChild(chns1,cids,fids,thrs,offset,k0,k);
			        h += hs[k]; if( h<=cascThr ) break;
			      }
			    } else if( treeDepth==2 ) {
			      // specialized case for treeDepth==2
			      for( int t = 0; t < nTrees; t++ ) {
			        uint32 offset=t*nTreeNodes, k=offset, k0=0;
			        getChild(chns1,cids,fids,thrs,offset,k0,k);
			        getChild(chns1,cids,fids,thrs,offset,k0,k);
			        h += hs[k]; if( h<=cascThr ) break;
			      }
			    } else if( treeDepth>2) {
			      // specialized case for treeDepth>2
			      for( int t = 0; t < nTrees; t++ ) {
			        uint32 offset=t*nTreeNodes, k=offset, k0=0;
			        for( int i=0; i<treeDepth; i++ )
			          getChild(chns1,cids,fids,thrs,offset,k0,k);
			        h += hs[k]; if( h<=cascThr ) break;
			      }
			    } else {
			      // general case (variable tree depth)
			      for( int t = 0; t < nTrees; t++ ) {
			        uint32 offset=t*nTreeNodes, k=offset, k0=k;
			        while( child[k] ) {
			          float ftr = chns1[cids[fids[k]]];
			          k = (ftr<thrs[k]) ? 1 : 0;
			          k0 = k = child[k0]-k+offset;
			        }
			        h += hs[k]; if( h<=cascThr ) break;
			      }
		    	}
			    if(h>cascThr) { cs.push_back(c); rs.push_back(r); hs1.push_back(h); }
		  	}
		}
		delete [] cids;
		free(chns);
		m=cs.size();

		// shift=(modelDsPad-modelDs)/2-pad;
		// double shift[2];
		// shift[0] = (modelHt-double(opts.modelDs[0]))/2-opts.pPyramid.pad[0];
		// shift[1] = (modelWd-double(opts.modelDs[1]))/2-opts.pPyramid.pad[1];

		for(int j=0; j<m; j++ )
		{
			BoundingBox bb = pyramidRowColumn2BoundingBox(rs[j], cs[j],  modelHt, modelWd, i, stride) ;

			// bb.topLeftPoint.x = cs[j]*stride;
			// bb.topLeftPoint.x = (bb.topLeftPoint.x+shift[1])/opts.pPyramid.scales_w[i];
			// bb.topLeftPoint.y = rs[j]*stride;
			// bb.topLeftPoint.y = (bb.topLeftPoint.y+shift[0])/opts.pPyramid.scales_h[i];
			// bb.height = opts.modelDs[0]/opts.pPyramid.scales[i];
			// bb.width = opts.modelDs[1]/opts.pPyramid.scales[i];
			bb.score = hs1[j];
			bb.scale = i;
			result.push_back(bb);
		}

		cs.clear();
		rs.clear();
		hs1.clear();
	}

	return result;
}

//bb = acfDetect1(P.data{i},Ds{j}.clf,shrink,modelDsPad(1),modelDsPad(2),opts.stride,opts.cascThr);
void Detector::acfDetect(std::vector<std::string> imageNames, std::string dataSetDirectoryName, int firstFrame, int lastFrame)
{
	int shrink = opts.pPyramid.pChns.shrink;
	int modelHt = opts.modelDsPad[0];
	int modelWd = opts.modelDsPad[1];
	int stride = opts.stride;
	float cascThr = opts.cascadeThreshold;

	cv::Mat tempThrs;
	cv::transpose(this->thrs, tempThrs);
	float *thrs = (float*)tempThrs.data;

	cv::Mat tempHs;
	cv::transpose(this->hs, tempHs);
	float *hs = (float*)tempHs.data;
	
	cv::Mat tempFids;
	cv::transpose(this->fids, tempFids);
	uint32 *fids = (uint32*) tempFids.data;
	
	cv::Mat tempChild;
	cv::transpose(this->child, tempChild);
	uint32 *child = (uint32*) tempChild.data;

	// const mwSize *fidsSize = mxGetDimensions(mxGetField(trees,0,"fids"));
	// const int nTreeNodes = (int) fidsSize[0];
 	// const int nTrees = (int) fidsSize[1];
	int nTreeNodes = this->fids.rows;
	int nTrees = this->fids.cols;
	
	int treeDepth = this->treeDepth;
	int nChns = opts.pPyramid.pChns.pColor.nChannels + opts.pPyramid.pChns.pGradMag.nChannels + opts.pPyramid.pChns.pGradHist.nChannels; 

	bool calibratedGetScalesDone = false;
	bool generateCandidatesDone = false;
	bool cidsDone = false;
	bool candidateRegionsDone = false;

	std::ofstream txtFile;
	if (config.saveDetectionsInText)
	{
		std::string outputfilename = config.outputFolder + "/detections.txt"; 
		txtFile.open (outputfilename.c_str());
	}

	int numberOfFrames=0;
	if (imageNames.size() <= lastFrame)
		numberOfFrames = imageNames.size()-firstFrame;
	else
		numberOfFrames = lastFrame-firstFrame+1;
	BB_Array_Array detections(numberOfFrames);

	BB_Array bbox_candidates;
	BB_Array denseCandidates;
	std::vector<uint32*> scales_cids;

	BB_Array_Array groundTruthDetections;
	if (config.useGroundTruth)
		//groundTruthDetections = readGroundTruth(config.groundTruthFileName, config.resizeImage);
		groundTruthDetections = readGroundTruthTopFile(config.groundTruthFileName, config.resizeImage);

	std::vector<int> numberOfCandidates; 
 
	for (int i = firstFrame; i < firstFrame + numberOfFrames; i++)
	{
		clock_t frameStart = clock();

		// this conversion is necessary, so we don't apply this transformation multiple times, which would break the image inside chnsPyramid
		cv::Mat image = cv::imread(dataSetDirectoryName + '/' + imageNames[i]);

		// if resizeImage is set different to 1.0, resize before computing the pyramid
		if (config.resizeImage != 1.0) 
			cv::resize(image, image, cv::Size(), config.resizeImage, config.resizeImage);

		cv::Mat I;
		// which one of these conversions is best?
		//image.convertTo(I, CV_32FC3, 1.0/255.0);
		cv::normalize(image, I, 0.0, 1.0, cv::NORM_MINMAX, CV_32FC3);

		std::vector<Info> framePyramid;
		BB_Array frameDetections;
		clock_t detectionStart;
		
		// decide if we should use the calibrated detector or the translated Dóllar detection
		if (config.useCalibration) 
		{
			// the decision of which scales are necessary is taken only on the first frame, since we assume the same camera for the whole data set
			if (!calibratedGetScalesDone)
			{
				opts.pPyramid.calibratedGetScales(I.rows, I.cols, shrink, modelWd, modelHt, config.maxPedestrianWorldHeight, *(config.projectionMatrix), *(config.homographyMatrix));
				calibratedGetScalesDone = true;

				std::cout << "Number of Scales: " << opts.pPyramid.scales.size() << " Last Scale: " << opts.pPyramid.scales[opts.pPyramid.scales.size()-1] << std::endl; 
				//std::cin.get();
			}

			// computes the feature pyramid for the current frame
			framePyramid = opts.pPyramid.computeFeaturePyramid(I, config.useCalibration); 

			// starts counting the time spent in detection for the current frame
 			detectionStart = clock();

 			// the cids matrix helps the classifier to access the pyramid patches and is calculated only once
 			if (!cidsDone)
 			{
 				scales_cids.reserve(opts.pPyramid.computedScales);
				for (int i=0; i < opts.pPyramid.computedScales; i++) 
				{
					int height = framePyramid[i].image.rows;
					int width = framePyramid[i].image.cols;
					int nFtrs = modelHt/shrink*modelWd/shrink*nChns;
				  	uint32 *cids = new uint32[nFtrs]; int m=0;
				  	
				  	for( int z=0; z<nChns; z++ ) {
				    	for( int cc=0; cc<modelWd/shrink; cc++ ) {
				      		for( int rr=0; rr<modelHt/shrink; rr++ ) {
				        		cids[m++] = z*width*height + cc*height + rr;
				        	}
				        }
				    }
				    scales_cids.push_back(cids);
				}
				cidsDone = true;
			}


			// the candidate patches are genereted only for the first frame, since we assume we have the same camera for the whole data set
 			if (!generateCandidatesDone)
 			{
	 			double maxHeight = 0; 

				if (config.candidateGeneration == FULL)
					bbox_candidates = generateCandidates(image.rows, image.cols, shrink, *(config.projectionMatrix), *(config.homographyMatrix), (float)opts.modelDs[1]/opts.modelDs[0]);
				else
					bbox_candidates = generateSparseCandidates(opts.modelDs[1], opts.modelDs[0], config.minPedestrianWorldHeight, config.maxPedestrianWorldHeight, image.cols, image.rows, shrink, *(config.projectionMatrix), *(config.homographyMatrix));

				generateCandidatesDone = true;
			}

			/*
			// pre-compute the way we access the features for each scale (needs to be done for every frame, since images are different)
			std::vector<float*> scales_chns(opts.pPyramid.computedScales, NULL);
			int imageHeights[opts.pPyramid.computedScales];
			int imageWidths[opts.pPyramid.computedScales];
			for (int j=0; j < opts.pPyramid.computedScales; j++) 
			{
				imageHeights[j] = framePyramid[j].image.rows;
				imageWidths[j] = framePyramid[j].image.cols;

				int channels = opts.pPyramid.pChns.pColor.nChannels + opts.pPyramid.pChns.pGradMag.nChannels + opts.pPyramid.pChns.pGradHist.nChannels;
				float* chns = (float*)malloc(imageHeights[j]*imageWidths[j]*channels*sizeof(float));
				features2floatArray(framePyramid[j], chns, imageHeights[j], imageWidths[j], opts.pPyramid.pChns.pColor.nChannels, opts.pPyramid.pChns.pGradMag.nChannels, opts.pPyramid.pChns.pGradHist.nChannels);
				scales_chns[j] = chns;
			}
			*/

			/*
			// debug
			showDetections(I, bbox_candidates, "Sparse Candidates", false);
			std::cout << bbox_candidates.size() << " sparse candidates\n";
			cv::waitKey();
			// debug */

			// aplies classifier to all candidate bounding boxes
 			//frameDetections = applyCalibratedDetectorToFrame(bbox_candidates, scales_chns, imageHeights, imageWidths, shrink, modelHt, modelWd, stride, cascThr, 
 			//				thrs, hs, scales_cids, fids, child, nTreeNodes, nTrees, treeDepth, nChns, image.cols, image.rows, *(config.projectionMatrix));
			frameDetections = applyCalibratedDetectorToFrame(framePyramid, bbox_candidates, shrink, modelHt, modelWd, stride, cascThr, 
 							thrs, hs, scales_cids, fids, child, nTreeNodes, nTrees, treeDepth, nChns, image.cols, image.rows, *(config.projectionMatrix));

			/*
			// debug
			showDetections(I, frameDetections, "Detections After First Step", false);
			cv::waitKey();
			// debug */

			if (config.candidateGeneration == SPARSE)
 			{
				clock_t candidateStart = clock();

				if (candidateRegions.size() == 0)
					denseCandidates = generateCandidateRegions(frameDetections, image.rows, image.cols, shrink, opts.modelDs[0], opts.modelDs[1], config.minPedestrianWorldHeight,
			 													config.maxPedestrianWorldHeight, *(config.projectionMatrix), *(config.homographyMatrix));
				else
				{
					BB_Array newCandidates = addCandidateRegions(frameDetections, image.rows, image.cols, opts.modelDs[0], opts.modelDs[1],
												config.minPedestrianWorldHeight, config.maxPedestrianWorldHeight, shrink, *(config.projectionMatrix), *(config.homographyMatrix));

					if (newCandidates.size() > 0)
						denseCandidates.insert(denseCandidates.end(), newCandidates.begin(), newCandidates.end());

					newCandidates.clear(); // is this necessary?
				} //
				clock_t candidateEnd = clock();

 				frameDetections.clear();

 				/*
 				// debug
				//showDetections(I, denseCandidates, "Dense Candidates", false);
				showDetections(I, candidateRegions, "Dense Regions", false);
				std::cout << denseCandidates.size() << " dense candidates\n";
				cv::waitKey();
				// debug */

 				frameDetections = applyCalibratedDetectorToFrame(framePyramid, denseCandidates, shrink, modelHt, modelWd, stride, cascThr, 
 							thrs, hs, scales_cids, fids, child, nTreeNodes, nTrees, treeDepth, nChns, image.cols, image.rows, *(config.projectionMatrix));

 				//frameDetections = applyCalibratedDetectorToFrame(denseCandidates, scales_chns, imageHeights, imageWidths, shrink, modelHt, modelWd, stride, cascThr, 
 				//			thrs, hs, scales_cids, fids, child, nTreeNodes, nTrees, treeDepth, nChns, image.cols, image.rows, *(config.projectionMatrix));
 				
 				/*
 				// debug
				showDetections(I, frameDetections, "Detections After Second Step", false);
				cv::waitKey();
				// debug */

				numberOfCandidates.push_back(bbox_candidates.size() + denseCandidates.size());

 				//remove dense regions where there was no detection
 				removeCandidateRegions(frameDetections, denseCandidates);
 				// */

 				/*
 				// debug
 				showDetections(I, candidateRegions, "Dense Regions", false);
				std::cout << denseCandidates.size() << " dense candidates\n";
				cv::waitKey();
 				// debug */
 			} // */

			// free the memory used to pre-allocate indexes
			//for (int i=0; i < opts.pPyramid.computedScales; i++) 
			//	free(scales_chns[i]);
		}
		else
		{
			// computes the feature pyramid for the current frame
			framePyramid = opts.pPyramid.computeFeaturePyramid(I, config.useCalibration);
			
			// starts counting the time spent in detection for the current frame
			detectionStart = clock();

			// aplies classifier to all image patches
			frameDetections = applyDetectorToFrame(framePyramid, shrink, modelHt, modelWd, stride, cascThr, thrs, hs, fids, child, nTreeNodes, nTrees, treeDepth, nChns);		
		}

		// saves the detections
		detections[i-firstFrame] = frameDetections;
		frameDetections.clear(); //doesn't seem to make a difference

		// finalizes the detection clock and adds the time spent to the total detection time
		clock_t detectionEnd = clock();
		timeSpentInDetection = timeSpentInDetection + (double(detectionEnd - detectionStart) / CLOCKS_PER_SEC);

		/*
		// debug: shows detections before suppression
		cv::imshow("source image", I);
		showDetections(I, detections[i], "detections before suppression", config.showScore);
		cv::waitKey();
		// debug */

		/*
		// debug: dispĺays the image pyramid for the current frame
		for (int s=0; s < framePyramid.size(); s++)
		{
			cv::imshow("Color Channel", framePyramid[s].image);
			cv::imshow("GradMag Channel", framePyramid[s].gradientMagnitude);
			cv::imshow("GradHist Channel", framePyramid[s].gradientHistogram[0]);
			cv::waitKey();
		}
		cv::destroyAllWindows();
		// debug */

		// decides which type of non-maximal suppression is used
		if (config.useCalibration)
		{
			if (config.candidateGeneration == SPARSE)
			{
				BB_Array tempDetections = nonMaximalSuppression(detections[i-firstFrame]);

				/*
				// debug
				showDetections(I, tempDetections, "Detections After First Suppression", false);
				cv::waitKey();
				// debug */

				detections[i-firstFrame] = ratedSuppression(tempDetections);

				/*
				// debug
				showDetections(I, detections[i-firstFrame], "Detections After Second Suppression", false);
				cv::waitKey();
				// debug */			
			}
			else 
			{ // full method
				int averagePedestrianWorldHeight = (config.maxPedestrianWorldHeight+config.minPedestrianWorldHeight)/2;
				detections[i-firstFrame] = nonMaximalSuppressionSmart(detections[i-firstFrame], averagePedestrianWorldHeight, averagePedestrianWorldHeight/5);
			}


		}
		else
			detections[i-firstFrame] = nonMaximalSuppression(detections[i-firstFrame]);

		//std::cout << "Total: " << groundTruthDetections[i].size() << std::endl;
		//std::cout << "Covered: " << findFramesCoveredPedestrians(groundTruthDetections[i],detections[i-firstFrame]) << std::endl;

		std::cout << "detections: " << detections[i-firstFrame].size() << std::endl;
		
		// shows detections after suppression
		if (config.displayDetections)
		{
			showDetections(I, detections[i-firstFrame], "detections after suppression", config.showScore);
			cv::waitKey(500);
		}	
		
		// saves image with embedded detections
		if (config.saveFrames) {
			for (int j = 0; j < detections[i-firstFrame].size(); j++) 
				detections[i-firstFrame][j].plot(image, cv::Scalar(0,255,0), config.showScore);

			/*
			for (int j = 0; j < candidateRegions.size(); j++)
				candidateRegions[j].plot(image, cv::Scalar(255,0,0), false);
			*/

			if (config.useGroundTruth)
				for (int j = 0; j < groundTruthDetections[i-firstFrame].size(); j++)
					groundTruthDetections[i][j].plot(image, cv::Scalar(255,0,0), false);

			std::string outputfilename = config.outputFolder + '/' + imageNames[i];
			cv::imwrite(outputfilename, image);
		}

		// saves the detections in text format
		if (config.saveDetectionsInText)
		{
  			for (int j = 0; j < detections[i-firstFrame].size(); j++)
  				txtFile << detections[i-firstFrame][j].toString(i);
  				//txtFile << detections[i-firstFrame][j].height << " " << detections[i-firstFrame][j].score << std::endl;
		}
		
		// clear memory occupied by the pyramid
		for (int j=0; j < opts.pPyramid.computedScales; j++)
		{
			framePyramid[j].image.release();
			framePyramid[j].gradientMagnitude.release();
			framePyramid[j].gradientHistogram.clear();
		}
		framePyramid.clear();
		image.release();
		I.release();

		// prints the total time spent working in the current frame
		clock_t frameEnd = clock();
		double elapsed_secs = double(frameEnd - frameStart) / CLOCKS_PER_SEC;
		if (config.useCalibration && config.candidateGeneration == SPARSE)
		{
			std::cout << "Frame " << i-firstFrame+1 << " of " << numberOfFrames << " was processed in " << elapsed_secs << " seconds "; 
			std::cout << "(" << bbox_candidates.size() + denseCandidates.size() << " candidates).\n";
		}
		else
			std::cout << "Frame " << i-firstFrame+1 << " of " << numberOfFrames << " was processed in " << elapsed_secs << " seconds.\n"; 
	}

	if (config.useCalibration)
	{
		for (int i=0; i < opts.pPyramid.computedScales; i++)
			free(scales_cids[i]);

		double candidatesPerFrame = std::accumulate(numberOfCandidates.begin(), numberOfCandidates.end(), 0.0)/numberOfCandidates.size();
		std::cout << "candidatesPerFrame=" << candidatesPerFrame << std::endl;
	}

	if (config.useGroundTruth)
	{
		int coveredPedestriansInDataSet=0;
		int totalPedestrians=0;
		int totalDetections=0;
		int totalFalsePositives;
		double coverageRatio, falsePositivesPerFrame;
		
		for(int i=0; i < detections.size(); i++)
		{
			coveredPedestriansInDataSet = coveredPedestriansInDataSet + findFramesCoveredPedestrians(groundTruthDetections[firstFrame+i],detections[i]);
			totalPedestrians = totalPedestrians + groundTruthDetections[firstFrame+i].size();
			totalDetections = totalDetections + detections[i].size();
		}
		
		totalFalsePositives = totalDetections - coveredPedestriansInDataSet;
		coverageRatio = (double)coveredPedestriansInDataSet / totalPedestrians;
		falsePositivesPerFrame = (double)totalFalsePositives / detections.size();

		std::cout << "\nCoverage: " << coverageRatio << "\nfalsePositivesPerFrame: " << falsePositivesPerFrame << std::endl;
	}

	if (config.saveDetectionsInText)
	{
		txtFile.close();
  	}
}

BB_Array Detector::ratedSuppression(BB_Array detections)
{
	std::vector<bool> discarded(detections.size(), false);
	std::sort(detections.begin(), detections.end());

	for (int i=detections.size()-1; i >= 0; i--)
	{
		for (int j=0; j < detections.size(); j++)
		{
			if (i != j && !discarded[i] && !discarded[j])
			{
				// calculate overlap
				double overlap = 0.0;
				double xei, xej, xmin, xsMax, iw;
				double yei, yej, ymin, ysMax, ih;
				xei = detections[i].topLeftPoint.x + detections[i].width;
				xej = detections[j].topLeftPoint.x + detections[j].width;
				xmin = xej;			
				if (xei < xej)
					xmin = xei;
				xsMax = detections[i].topLeftPoint.x;
				if (detections[j].topLeftPoint.x > detections[i].topLeftPoint.x)
					xsMax = detections[j].topLeftPoint.x;
				iw = xmin - xsMax;
				yei = detections[i].topLeftPoint.y + detections[i].height;
				yej = detections[j].topLeftPoint.y + detections[j].height;
				ymin = yej;			
				if (yei < yej)
					ymin = yei;
				ysMax = detections[i].topLeftPoint.y;
				if (detections[j].topLeftPoint.y > detections[i].topLeftPoint.y)
					ysMax = detections[j].topLeftPoint.y;
				ih = ymin - ysMax;
				if (iw  > 0 && ih > 0)
				{
					overlap = iw * ih;
					double u = detections[i].height*detections[i].width + detections[j].height*detections[j].width-overlap;
					//u = detections[i].height*detections[i].width;
					//if (detections[i].height*detections[i].width > detections[j].height*detections[j].width)
					//	u = detections[j].height*detections[j].width;
					overlap = overlap/u;
				}

				// calculate the ratio between the two bounding boxes' scores
				float scoreRatio;
				if (detections[j].score > detections[i].score)
					scoreRatio = detections[i].score / detections[j].score;
				else
					scoreRatio = detections[j].score / detections[i].score;

				if (overlap > scoreRatio)
				{
					if (detections[i].score < detections[j].score)
						discarded[i] = true;
					else
						discarded[j] = true;
				}
			}
		}
	}

	BB_Array result;
	int resultIndex=0;
	for (int i=0; i < discarded.size(); i++)
		if (!discarded[i])
			result.push_back(detections[i]);

	return result;
}

// for each i suppress all j st j>i and area-overlap>overlap
BB_Array nmsMax(BB_Array source, bool greedy, double overlapArea, cv::String overlapDenominator)
{
	BB_Array sortedArray(source.size());
	bool *discarded = (bool*)malloc(source.size()*sizeof(bool));
	int discardedBBs = 0;

	for (int i=0; i < source.size(); i++)
	{
		sortedArray[i] = source[i];
		discarded[i] = false;
	}
 
	std::sort(sortedArray.begin(), sortedArray.end());
	
	for (int i = 0; i < sortedArray.size(); i++)
	{
		if (!greedy || !discarded[i]) // continue only if its not greedy or result[i] was not yet discarded
		{
			for (int j = i+1; j < sortedArray.size(); j++)
			{
				if (discarded[j] == false) // continue this iteration only if result[j] was not yet discarded
				{
					double xei, xej, xmin, xsMax, iw;
					double yei, yej, ymin, ysMax, ih;
					xei = sortedArray[i].topLeftPoint.x + sortedArray[i].width;
					xej = sortedArray[j].topLeftPoint.x + sortedArray[j].width;
					xmin = xej;			
					if (xei < xej)
						xmin = xei;
					xsMax = sortedArray[i].topLeftPoint.x;
					if (sortedArray[j].topLeftPoint.x > sortedArray[i].topLeftPoint.x)
						xsMax = sortedArray[j].topLeftPoint.x;
					iw = xmin - xsMax;
					yei = sortedArray[i].topLeftPoint.y + sortedArray[i].height;
					yej = sortedArray[j].topLeftPoint.y + sortedArray[j].height;
					ymin = yej;			
					if (yei < yej)
						ymin = yei;
					ysMax = sortedArray[i].topLeftPoint.y;
					if (sortedArray[j].topLeftPoint.y > sortedArray[i].topLeftPoint.y)
						ysMax = sortedArray[j].topLeftPoint.y;
					ih = ymin - ysMax;
					if (iw  > 0 && ih > 0)
					{
						double o = iw * ih;
						double u;
						if (overlapDenominator == "union")
							u = sortedArray[i].height*sortedArray[i].width + sortedArray[j].height*sortedArray[j].width-o;
						else if (overlapDenominator == "min")
						{
							u = sortedArray[i].height*sortedArray[i].width;
							if (sortedArray[i].height*sortedArray[i].width > sortedArray[j].height*sortedArray[j].width)
								u = sortedArray[j].height*sortedArray[j].width;
						}
						o = o/u;
						if (o > overlapArea) // sortedArray[j] is no longer needed (is discarded)
						{
							discarded[j] = true;
							discardedBBs++;
						}
					}
				}
			}	
		}
	}
	
	BB_Array result(source.size()-discardedBBs);
	int resultIndex=0;
	// result keeps only the bounding boxes that were not discarded
	for (int i=0; i < sortedArray.size(); i++)
		if (!discarded[i])
			result[resultIndex++] = sortedArray[i];

	free(discarded);

	return result;
}

BB_Array Detector::nonMaximalSuppression(BB_Array bbs)
{
	BB_Array result;

	//keep just the bounding boxes with scores higher than the threshold
	for (int i=0; i < bbs.size(); i++)
		if (bbs[i].score > config.supressionThreshold)
			result.push_back(bbs[i]);

	// bbNms would apply resize to the bounding boxes now
	// but our models dont use that, so it will be suppressed
		
	// since we just apply one detector model at a time,
	// our separate attribute would always be false
	// so the next part is simpler, nms1 follows
	
	// if there are too many bounding boxes,
	// he splits into two arrays and recurses, merging afterwards
	// this will be done if necessary
	
	// run actual nms on given bbs
	// other types might be added later
	switch (opts.suppressionType)
	{
		case MAX:
			result = nmsMax(result, false, opts.overlapArea, opts.overlapDenominator);
		break;
		case MAXG:
			result = nmsMax(result, true, opts.overlapArea, opts.overlapDenominator);
		break;
		case MS:
			// not yet implemented
		break;
		case COVER:
			// not yet implemented
		break;	
	}

	return result;
}

BB_Array Detector::nonMaximalSuppressionSmart(BB_Array bbs, double meanHeight, double stdHeight)
{
	BB_Array result;

	// keep just the bounding boxes with scores higher than the threshold
	// for (int i=0; i < bbs.size(); i++)
	// 	if (bbs[i].score > opts.suppressionThreshold)
	// 		result.push_back(bbs[i]);

	for (int i=0; i < bbs.size(); ++i) {
		bbs[i].score = bbs[i].score*gaussianFunction(meanHeight, stdHeight, bbs[i].worldHeight);
		if (bbs[i].score > config.supressionThreshold)
			result.push_back(bbs[i]);
	}


	// run actual nms on given bbs
	// other types might be added later
	switch (opts.suppressionType)
	{
		case MAX:
			result = nmsMax(result, false, opts.overlapArea, opts.overlapDenominator);
		break;
		case MAXG:
			result = nmsMax(result, true, opts.overlapArea, opts.overlapDenominator);
		break;
		case MS:
			// not yet implemented
		break;
		case COVER:
			// not yet implemented
		break;	
	}

	return result;
}