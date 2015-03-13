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

		if (settings.firstFrame > imageNames.size())
			settings.firstFrame = 0;

		if (settings.lastFrame < settings.firstFrame)
			settings.lastFrame = imageNames.size();

		
		cv::Mat image = cv::imread(settings.dataSetDirectory + '/' + imageNames[0]);

		/*
		BB_Array testBoxes;
		testBoxes.push_back(BoundingBox(871, 400, 100, 218));
		testBoxes.push_back(BoundingBox(0, 0, 41, 100));
		testBoxes.push_back(BoundingBox(1000, 0, 41, 100));
		testBoxes.push_back(BoundingBox(1800, 0, 41, 100));
		testBoxes.push_back(BoundingBox(0, 800, 41, 100));
		for (int i=0; i < testBoxes.size(); i++)
		{
			testBoxes[i].plot(image, cv::Scalar(0,255,0));
			double height = findWorldHeight(testBoxes[i].topLeftPoint.x, testBoxes[i].topLeftPoint.y+testBoxes[i].height, testBoxes[i].topLeftPoint.y, (*settings.projectionMatrix), (*settings.homographyMatrix));
			std::cout << "BB" << i+1 << " World Height: " << height << std::endl;
		}
		imshow("test", image);
		cv::waitKey();
		// */
		
		// apply the detection on all images
		d.acfDetect(imageNames, settings.dataSetDirectory, settings.firstFrame, settings.lastFrame);

		//std::cout << "before printing final times\n";
		//std::cin.get();

		clock_t end = clock();
		double elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
		std::cout << "\nTotal processing time was " << elapsed_secs << " seconds.\n";
		std::cout << "Time elapsed calculating features: " << d.opts.pPyramid.totalTimeForRealScales << std::endl;
		std::cout << "Time elapsed approximating features: " << d.opts.pPyramid.totalTimeForApproxScales << std::endl;
		std::cout << "Time elapsed during detections: " << d.timeSpentInDetection << std::endl;

		//std::cout << "after printing final times\n";
		//std::cin.get();

		delete settings.projectionMatrix;
		delete settings.homographyMatrix;

		return 0;
	}
}
