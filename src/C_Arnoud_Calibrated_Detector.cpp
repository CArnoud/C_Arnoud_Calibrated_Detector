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

		// apply the detection on all images
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
