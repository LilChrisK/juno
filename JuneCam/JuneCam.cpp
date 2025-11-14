// JuneCam.cpp : This file contains the 'main' function. Program execution begins and ends there.

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

int main(int argc, char** argv) {
	namespace fs = std::filesystem;
	std::cout << "Working directory: " << fs::current_path().string() << "\n";

	std::string fname = "images/raw/JNCE_2022056_40C00036_V01-raw.png";

	cv::Mat raw = cv::imread(fname, cv::IMREAD_UNCHANGED);
	if (raw.empty()) {
		std::cerr << "Could not open raw image: " << fname << "\n";
		return 1;
	}

	int width = raw.cols;
	int rows = raw.rows;
	std::cout << "Raw size: " << width << " x " << rows << "\n";

	// Adjust these parameters for your raw file
	int bandHeight = 128;   // strips height for JunoCam visible filters
	int bands = 3;          // we are using R, G, B (ignoring methane for now)
	// Offsets (assuming first strip is Blue, then Green, then Red)

	int redOffset = 2 * bandHeight;
	int greenOffset = bandHeight;
	int blueOffset = 0;

	int frames = rows / (bandHeight * bands);
	std::cout << "Frames count: " << frames << "\n";

	// Create mosaic per channel
	cv::Mat redMosaic = cv::Mat::zeros(frames * bandHeight, width, raw.type());
	cv::Mat blueMosaic = cv::Mat::zeros(frames * bandHeight, width, raw.type());
	cv::Mat greenMosaic = cv::Mat::zeros(frames * bandHeight, width, raw.type());


	for (int f = 1; f < frames - 1; ++f) {
		int baseRow = f * bandHeight * bands;

		// Red
		cv::Mat stripR = raw(cv::Range(baseRow + redOffset,
			baseRow + redOffset + bandHeight),
			cv::Range::all());
		stripR.copyTo(redMosaic(cv::Range(f * bandHeight + bandHeight,
			(f + 1) * bandHeight + bandHeight),
			cv::Range::all()));

		// Green
		cv::Mat stripG = raw(cv::Range(baseRow + greenOffset,
			baseRow + greenOffset + bandHeight),
			cv::Range::all());
		stripG.copyTo(greenMosaic(cv::Range(f * bandHeight,
			(f + 1) * bandHeight),
			cv::Range::all()));

		// Blue
		cv::Mat stripB = raw(cv::Range(baseRow + blueOffset,
			baseRow + blueOffset + bandHeight),
			cv::Range::all());
		stripB.copyTo(blueMosaic(cv::Range(f * bandHeight - bandHeight,
			(f + 1) * bandHeight - bandHeight),
			cv::Range::all()));
	}

	// Write single-channel mosaics if desired
	cv::imwrite("images/processed/red_channel.png", redMosaic);
	cv::imwrite("images/processed/green_channel.png", greenMosaic);
	cv::imwrite("images/processed/blue_channel.png", blueMosaic);
	std::cout << "Single-channel mosaics written.\n";

	// Convert to 8-bit if raw type is different (just in case)
	cv::Mat red8, green8, blue8;
	cv::normalize(redMosaic, red8, 0, 255, cv::NORM_MINMAX);
	cv::normalize(greenMosaic, green8, 0, 255, cv::NORM_MINMAX);
	cv::normalize(blueMosaic, blue8, 0, 255, cv::NORM_MINMAX);

	red8.convertTo(red8, CV_8UC1);
	green8.convertTo(green8, CV_8UC1);
	blue8.convertTo(blue8, CV_8UC1);

	// Merge into an RGB image (OpenCV uses BGR by default, so put Blue first)
	std::vector<cv::Mat> channels = { blue8, green8, red8 };
	cv::Mat rgbMosaic;
	cv::merge(channels, rgbMosaic);

	cv::imwrite("images/processed/combined_rgb.png", rgbMosaic);
	std::cout << "Combined RGB image written (combined_rgb.png).\n";

	return 0;
}
