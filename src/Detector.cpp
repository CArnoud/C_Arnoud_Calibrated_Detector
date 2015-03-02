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
		std::cout << "Configuration file named " << config_file << " was not found.\n"; 
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
// for now, it must be like this since the current model was not written by this program 
// this will change after we are set on a class structure
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

void showDetections(cv::Mat I, BB_Array detections, cv::String windowName)
{
	cv::Mat img = I.clone();
	for (int j = 0; j<detections.size(); j++) 
		detections[j].plot(img, cv::Scalar(0,255,0));

	cv::imshow(windowName, img);
}

void printDetections(BB_Array detections, int frameIndex)
{
	std::cout << "Detections in frame " << frameIndex << ":\n";
	for (int i=0; i < detections.size(); i++)
		std::cout << detections[i].toString(frameIndex) << std::endl;
	std::cout << std::endl;
}

// this procedure was just copied verbatim
inline void getChild(float *chns1, uint32 *cids, uint32 *fids, float *thrs, uint32 offset, uint32 &k0, uint32 &k)
{
  float ftr = chns1[cids[fids[k]]];
  k = (ftr<thrs[k]) ? 1 : 2;
  k0=k+=k0*2; k+=offset;
}

BB_Array* Detector::generateCandidateRegions(BB_Array* candidates, int imageHeight, int imageWidth, int shrink, int modelHeight, int modelWidth, 
											float minPedestrianHeight, float maxPedestrianHeight, cv::Mat_<float> &P, cv::Mat_<float> &H)
{
	BB_Array* result = new BB_Array();
	BB_Array tempResult;

	// sort the candidates by score
	BB_Array sortedCandidates(candidates->size());
	for (int i=0; i < candidates->size(); i++)
	{
		sortedCandidates[i] = (*candidates)[i];
	} 
	std::sort(sortedCandidates.begin(), sortedCandidates.begin()+sortedCandidates.size());

	// keeps only the biggest bounding box of each region (could be relocated to the classifier)
	for (int i = 0; i < candidates->size(); ++i)
	{
		bool discard = false;
		int j=0;
		while (j < tempResult.size() && !discard)
		{
			if (sortedCandidates[i].topLeftPoint.x >= tempResult[j].topLeftPoint.x-tempResult[j].width &&
				sortedCandidates[i].topLeftPoint.x <= tempResult[j].topLeftPoint.x+tempResult[j].width &&
				sortedCandidates[i].topLeftPoint.y >= tempResult[j].topLeftPoint.y-(tempResult[j].height/2) &&
				sortedCandidates[i].topLeftPoint.y <= tempResult[j].topLeftPoint.y+(tempResult[j].height/2)) 
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
		{
			tempResult.push_back(sortedCandidates[i]);
		}
	}

	//std::cout << tempResult.size() << " regions to be created\n";

	// here, we need to create the regions around the remaining bounding boxes
	for (int i=0; i < tempResult.size(); i++)
	{
		//std::cout << "basePoint: (" << tempResult[i].topLeftPoint.x << "," << tempResult[i].topLeftPoint.y << "), height=" <<
		//tempResult[i].height << ", width=" << tempResult[i].width << std::endl;

		int v = tempResult[i].topLeftPoint.y + (tempResult[i].height/2);
		int maxV = tempResult[i].topLeftPoint.y + (3*tempResult[i].height/2);
		if (maxV > imageHeight)
			maxV = imageHeight;

		bool foo=false;

		while (v < maxV)
		{
			int u = tempResult[i].topLeftPoint.x - tempResult[i].width;
			int maxU = tempResult[i].topLeftPoint.x + tempResult[i].width;
			if (u < 0)
				u = 0;
			if (maxU > imageWidth-modelWidth)
				maxU = imageWidth-modelWidth;

			/*
			if (!foo)
			{
				std::cout << "minV=" << v << ", maxV=" << maxV << ", minU=" << u << ", maxU=" << maxU << std::endl;
				foo = true;
			}
			*/

			while (u < maxU)
			{
				// we start at the top of the region
				int head_v = tempResult[i].topLeftPoint.y - (tempResult[i].height/2); 
				
				double bbWorldHeight = findWorldHeight(u, v, head_v, P, H);
				while (head_v < v-modelHeight && bbWorldHeight > maxPedestrianHeight)
				{
					head_v++;
					bbWorldHeight = findWorldHeight(u, v, head_v, P, H);
				}

				if (bbWorldHeight <= maxPedestrianHeight)
				{
					// found the biggest valid bounding box in the point
					int bbHeight = v-head_v;
					int bbWidth = bbHeight*modelWidth/modelHeight;
					int bbScale = findClosestScaleFromBbox(bbHeight, imageHeight);
					BoundingBox maxCandidate(u, head_v, bbWidth, bbHeight, bbScale, bbWorldHeight);
					result->push_back(maxCandidate);

					int previousScale = -1;

					// now, find other valid bounding boxes
					while (bbWorldHeight > minPedestrianHeight && head_v < v-modelHeight)
					{
						head_v++;
						bbWorldHeight = findWorldHeight(u, v, head_v, P, H);

						if (bbWorldHeight >= minPedestrianHeight)
						{
							bbHeight = v-head_v;
							bbScale = findClosestScaleFromBbox(bbHeight, imageHeight);

							if (bbScale != previousScale)
							{
								previousScale = bbScale;

								bbWidth = bbHeight*modelWidth/modelHeight;
								BoundingBox newCandidate(u, head_v, bbWidth, bbHeight, bbScale, bbWorldHeight);
								result->push_back(newCandidate);
							}
						}
					}
				}
				u = u + shrink;
			}
			v++;
		}

		/*
		std::cout << result->size() << " candidates created so far.\n";
		std::cin.get();
		*/
	}

	return result;
} 

BB_Array* Detector::generateSparseCandidates(int modelWidth, int modelHeight, float minPedestrianHeight, float maxPedestrianHeight, int imageWidth, 
											int imageHeight, cv::Mat_<float> &P, cv::Mat_<float> &H) 
{
	int u;
	int v = modelHeight;
	int vStep;
	BB_Array *candidates = new BB_Array();

	while (v < imageHeight)
	{
		int sumBBHeights = 0;
		//int boundingBoxesOnRow=0;
		u=0;

		while (u < imageWidth-modelWidth)
		{
			int head_v = 0;
			int uStep;
			double bbWorldHeight = findWorldHeight(u, v, head_v, P, H);

			while (head_v < v-modelHeight && bbWorldHeight > maxPedestrianHeight)
			{
				head_v++;
				bbWorldHeight = findWorldHeight(u, v, head_v, P, H);
			}

			if (bbWorldHeight <= maxPedestrianHeight)
			{
				// found the biggest valid bounding box in the point
				int bbHeight = v-head_v;
				int bbWidth = bbHeight*modelWidth/modelHeight;
				int bbScale = findClosestScaleFromBbox(bbHeight, imageHeight);
				BoundingBox maxCandidate(u, head_v, bbWidth, bbHeight, bbScale, bbWorldHeight);
				candidates->push_back(maxCandidate);

				sumBBHeights = sumBBHeights + bbHeight;
				//boundingBoxesOnRow++;

				// now, find other valid bounding boxes
				while (bbWorldHeight > minPedestrianHeight && head_v < v-modelHeight)
				{
					head_v++;
					bbWorldHeight = findWorldHeight(u, v, head_v, P, H);

					if (bbWorldHeight >= minPedestrianHeight)
					{
						bbHeight = v-head_v;
						bbWidth = bbHeight*modelWidth/modelHeight;
						bbScale = findClosestScaleFromBbox(bbHeight, imageHeight);
						BoundingBox newCandidate(u, head_v, bbWidth, bbHeight, bbScale, bbWorldHeight);
						candidates->push_back(newCandidate);

						sumBBHeights = sumBBHeights + bbHeight;
						//boundingBoxesOnRow++;
					}
				}

				// whould we step half the smallest valid bounding box width to the right or use the smallest possible bounding box?
				// uStep = floor(bbWidth/2);
			}

			//if (uStep == 0)
			uStep = floor(modelWidth/2);
			u = u + uStep;
		}

		// if we found at least one bounding box in the row we step half the average bounding box height down
		//if (boundingBoxesOnRow > 0)
		//	vStep = floor(sumBBHeights/boundingBoxesOnRow);
		//else
		vStep = floor(modelHeight/2);

		v = v + vStep;
	}

	return candidates;
}

BB_Array* Detector::generateCandidates(int imageHeight, int imageWidth, int shrink, cv::Mat_<float> &P, cv::Mat_<float> &H, float BBwidth2heightRatio, 
										float meanHeight/* = 1.7m*/, float stdHeight/* = 0.1m*/, float factorStdHeight/* = 2.0*/) 
{

	// there is a set of parameters here that are hard coded, but should
	// be read from a file or something...
	cv::Mat_<float> P3 = P.col(2);

	float aspectRatio = BBwidth2heightRatio;
	float minImageHeight = 80;

	float stepHeight = 100;
	int totalCandidates = 0;

	BB_Array *candidates = new BB_Array();
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
						candidates->push_back(bb);
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

void Detector::bbTopLeft2PyramidRowColumn(int *r, int *c, BoundingBox &bb, int modelHt, int modelWd, int ith_scale, int stride) {


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

BoundingBox Detector::pyramidRowColumn2BoundingBox(int r, int c,  int modelHt, int modelWd, int ith_scale, int stride) {

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

BB_Array Detector::applyCalibratedDetectorToFrame(BB_Array* bbox_candidates, std::vector<float*> scales_chns, int *imageHeigths, int *imageWidths, int shrink, 
											int modelHt, int modelWd, int stride, float cascThr, float *thrs, float *hs, std::vector<uint32*> scales_cids, 
											uint32 *fids, uint32 *child, int nTreeNodes, int nTrees, int treeDepth, int nChns, int imageWidth, int imageHeight, 
											cv::Mat_<float> &P, cv::Mat &debug_image)
{
	BB_Array result;

	float max_h = -1000;

	for (int i = 0; i < bbox_candidates->size(); ++i) {
		// see which scale is best suited to the candidate
		int ith_scale = (*bbox_candidates)[i].scale;
		
		int height = imageHeigths[ith_scale];                                                              
		int width = imageWidths[ith_scale];

		// r and c are defined by the candidate itself
		int r, c;
		bbTopLeft2PyramidRowColumn(&r, &c, (*bbox_candidates)[i], modelHt, modelWd, ith_scale, stride);
		
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

	    if (h > max_h) {
	    	max_h = h;
	    }

	    double hf = h*gaussianFunction(1800, 300, (*bbox_candidates)[i].worldHeight);
	    //if(hf>config.classifierThreshold){
	    if(hf>1.0){
			// std::cout << h << std::endl;
			// std::cout << "hey" << std::endl;
			//cv::imshow("results", debug_image);
			BoundingBox detection((*bbox_candidates)[i]);
			detection.score = hf;
			detection.scale = ith_scale;

	    	result.push_back(detection);
	    	// bbox_candidates[i].plot(debug_image, cv::Scalar(0, 255, 0));
	    	//cv::waitKey(100);
	    }
	
	}

	return result;
}

BB_Array Detector::applyDetectorToFrame(std::vector<Info> pyramid, int shrink, int modelHt, int modelWd, int stride, float cascThr, float *thrs, float *hs, 
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

	BB_Array *bbox_candidates;
	std::vector<uint32*> scales_cids;

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
		
		if (config.useCalibration) // decides if we use the calibrated detector or just the Dóllar detection
		{
			// the decision of which scales are necessary is taken only on the first frame, since we assume the same camera for the whole data set
			if (!calibratedGetScalesDone)
			{
				opts.pPyramid.calibratedGetScales(I.rows, I.cols, shrink, modelWd, modelHt, config.maxPedestrianWorldHeight, *(config.projectionMatrix), *(config.homographyMatrix));
				calibratedGetScalesDone = true;
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

				// should we use modelDs or modelHt/modelWd?
				if (config.candidateGeneration == FULL)
					bbox_candidates = generateCandidates(image.rows, image.cols, shrink, *(config.projectionMatrix), *(config.homographyMatrix), (float)opts.modelDs[1]/opts.modelDs[0]);
				else
					bbox_candidates = generateSparseCandidates(opts.modelDs[1], opts.modelDs[0], config.minPedestrianWorldHeight, config.maxPedestrianWorldHeight, image.cols, image.rows, *(config.projectionMatrix), *(config.homographyMatrix));

				//std::cout << (*bbox_candidates).size() << " candidates generated on first step\n";

				/*
				// debug: shows the candidates
				showDetections(I, (*bbox_candidates), "candidates");
				cv::waitKey();
				// debug */

				generateCandidatesDone = true;
			}

			// pre-compute the way we access the features for each scale (needs to be done for every frame, since images are different)
			std::vector<float*> scales_chns(opts.pPyramid.computedScales, NULL);
			int imageHeights[opts.pPyramid.computedScales];
			int imageWidths[opts.pPyramid.computedScales];
			for (int i=0; i < opts.pPyramid.computedScales; i++) 
			{
				imageHeights[i] = framePyramid[i].image.rows;
				imageWidths[i] = framePyramid[i].image.cols;

				int channels = opts.pPyramid.pChns.pColor.nChannels + opts.pPyramid.pChns.pGradMag.nChannels + opts.pPyramid.pChns.pGradHist.nChannels;
				float* chns = (float*)malloc(imageHeights[i]*imageWidths[i]*channels*sizeof(float));
				features2floatArray(framePyramid[i], chns, imageHeights[i], imageWidths[i], opts.pPyramid.pChns.pColor.nChannels, opts.pPyramid.pChns.pGradMag.nChannels, opts.pPyramid.pChns.pGradHist.nChannels);
				scales_chns[i] = chns;
			}

			// aplies classifier to all candidate bounding boxes
 			frameDetections = applyCalibratedDetectorToFrame(bbox_candidates, scales_chns, imageHeights, imageWidths, shrink, modelHt, modelWd, stride, cascThr, 
 							thrs, hs, scales_cids, fids, child, nTreeNodes, nTrees, treeDepth, nChns, image.cols, image.rows, *(config.projectionMatrix), image);
		
 			if (config.candidateGeneration == SPARSE)
 			{
 				clock_t candidateStart = clock();

 				BB_Array *newCandidates = generateCandidateRegions(&frameDetections, image.rows, image.cols, shrink, opts.modelDs[0], opts.modelDs[1], config.minPedestrianWorldHeight,
 				 													config.maxPedestrianWorldHeight, *(config.projectionMatrix), *(config.homographyMatrix));
 				clock_t candidateEnd = clock();


 				//std::cout << newCandidates->size() << " candidates generated in " << (double(candidateEnd - candidateStart) / CLOCKS_PER_SEC) << " seconds\n"; 
 				
 				/*
				// debug: shows the candidate regions
				showDetections(I, (*newCandidates), "candidates");
				cv::waitKey();
				// debug */

 				frameDetections.clear();

 				frameDetections = applyCalibratedDetectorToFrame(newCandidates, scales_chns, imageHeights, imageWidths, shrink, modelHt, modelWd, stride, cascThr, 
 							thrs, hs, scales_cids, fids, child, nTreeNodes, nTrees, treeDepth, nChns, image.cols, image.rows, *(config.projectionMatrix), image);
 			}

			// free the memory used to pre-allocate indexes
			for (int i=0; i < opts.pPyramid.computedScales; i++) 
				free(scales_chns[i]);
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
		showDetections(I, detections[i], "detections before suppression");
		cv::waitKey();
		// debug */

		// decides which type of non-maximal suppression is used
		if (config.useCalibration)
			detections[i-firstFrame] = nonMaximalSuppressionSmart(detections[i-firstFrame], 1800, 100);
		else
			detections[i-firstFrame] = nonMaximalSuppression(detections[i-firstFrame]);
		
		// shows detections after suppression
		if (config.displayDetections)
		{
			showDetections(I, detections[i-firstFrame], "detections after suppression");
			//printDetections(detections[i], i);
			cv::waitKey(500);
		}		
		
		// saves image with embedded detections
		if (config.saveFrames) {
			for (int j = 0; j<detections[i-firstFrame].size(); j++) 
				detections[i-firstFrame][j].plot(image, cv::Scalar(0,255,0));

			std::string outputfilename = config.outputFolder + '/' + imageNames[i];
			cv::imwrite(outputfilename, image);
		}

		if (config.saveDetectionsInText)
		{
  			for (int j = 0; j < detections[i-firstFrame].size(); j++)
  				txtFile << detections[i-firstFrame][j].toString(i);
		}
		
		// experimental: do i need to clear these?
		for (int j=0; j < opts.pPyramid.computedScales; j++)
		{
			framePyramid[j].image.release();
			framePyramid[j].gradientMagnitude.release();
			framePyramid[j].gradientHistogram.clear();
		}
		image.release();
		I.release();
		// experimental */

		// prints the total time spent working in the current frame
		clock_t frameEnd = clock();
		double elapsed_secs = double(frameEnd - frameStart) / CLOCKS_PER_SEC;
		std::cout << "Frame " << i-firstFrame+1 << " of " << numberOfFrames << " was processed in " << elapsed_secs << " seconds.\n"; 
	}

	if (config.useCalibration)
		delete bbox_candidates;

	if (config.saveDetectionsInText)
	{
		txtFile.close();
  	}
}

// for each i suppress all j st j>i and area-overlap>overlap
BB_Array nmsMax(BB_Array source, bool greedy, double overlapArea, cv::String overlapDenominator)
{
	BB_Array sortedArray(source.size());
	// bool discarded[source.size()];
	bool *discarded = (bool*)malloc(source.size()*sizeof(bool));
	int discardedBBs = 0;

	for (int i=0; i < source.size(); i++)
	{
		sortedArray[i] = source[i];
		discarded[i] = false;
	}
 
	std::sort(sortedArray.begin(), sortedArray.begin()+sortedArray.size());
	
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
					if (iw > 0 && ih > 0)
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