#include <sstream>
#include "C_Arnoud_Calibrated_Detector.h"

// memory check: 	valgrind --tool=memcheck --leak-check=full --log-file=valgrind.log --error-limit=no ./C_Arnoud_Calibrated_Detector ../C_Arnoud_Calibrated_Detector/PETS.conf
// profiler:		valgrind --tool=callgrind ./C_Arnoud_Calibrated_Detector ../C_Arnoud_Calibrated_Detector/PETS.conf
// example call: 	./C_Arnoud_Calibrated_Detector ../C_Arnoud_Calibrated_Detector/PETS.conf
int main(int argc, char *argv[]) 
{
	if (argc < 2)
	{
		std::cout << " # Argument Error: this program requires a conf file." << std::endl;
		return 1;
	}
	else
	{
		clock_t start = clock();
		
		Config settings(argv[1]);
		Detector d(settings);

		// loads all detector settings from the provided xml file
		d.importDetectorModel(settings.detectorFileName);

		// gets names for all the files inside the data set folder
		std::vector<std::string> imageNames = getDataSetFileNames(settings.dataSetDirectory);

		// adjusts values of firstFrame and lastFrame to avoid errors
		if (settings.firstFrame > imageNames.size())
			settings.firstFrame = 0;
		if (settings.lastFrame < settings.firstFrame)
			settings.lastFrame = imageNames.size();

		/*
		// debug
		std::ifstream in_file;
		in_file.open("../datasets/PETS09_View001_S2_L1_000to794.avi.detection.modified.xml");

		if (in_file.is_open())
		{
			std::string token;
			BB_Array_Array groundTruth;
			while (in_file >> token && token != "</dataset>")
			{
				BB_Array frame1BBs, resizedBBs;
				while (in_file >> token && token != "</frame>") {
					if (token == "<box>") 
					{
						float height, width, x, y;
						while(token != "</box>")
						{
							in_file >> token;
							if (token == "<height>")
								in_file >> height;						
							if (token == "<width>")
								in_file >> width;
							if (token == "<x>")
								in_file >> x;
							if (token == "<y>")
								in_file >> y;
						}
						BoundingBox newBB(x, y, width, height);
						frame1BBs.push_back(newBB);
						newBB.resize(settings.resizeImage);
						resizedBBs.push_back(newBB);
					}
				}

				groundTruth.push_back(frame1BBs);

				cv::Mat image = cv::imread(settings.dataSetDirectory + '/' + imageNames[groundTruth.size()-1]);
				cv::Mat resImage;
				cv::resize(image, resImage, cv::Size(), settings.resizeImage, settings.resizeImage);
				
				for (int i=0; i < frame1BBs.size(); i++)
				{
					frame1BBs[i].plot(image, cv::Scalar(0,255,0), false);
					resizedBBs[i].plot(resImage, cv::Scalar(0,255,0), false);
				}

				cv::imshow("testing ground truth", image);
				cv::imshow("testing resized ground truth", resImage);
				cv::waitKey();
			}
		}
		else
			std::cout << "Cant find ground Truth\n";
		// debug */

		// applies the detection on all images
		d.acfDetect(imageNames, settings.dataSetDirectory, settings.firstFrame, settings.lastFrame);

		clock_t end = clock();
		double elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
		std::cout << "\nTotal processing time was " << elapsed_secs << " seconds.\n";
		std::cout << "Time elapsed calculating features: " << d.opts.pPyramid.totalTimeForRealScales << std::endl;
		std::cout << "Time elapsed approximating features: " << d.opts.pPyramid.totalTimeForApproxScales << std::endl;
		std::cout << "Time elapsed during detections: " << d.timeSpentInDetection << std::endl;

		delete settings.projectionMatrix;
		delete settings.homographyMatrix;

		return 0;
	}
}
