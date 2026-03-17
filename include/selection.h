#ifndef SELECTION_H
#define SELECTION_H

// CPP STDLIB

#include <chrono>
#include <vector>
#include <algorithm>

// UNORIGINAL SUBROUTINES

#include <fmt/base.h>
#include <fmt/ranges.h>

// ORIGINAL CODE

#include "data.h"
#include "nn.h"

void forward_selection(const Dataset& dataset) {
	fmt::print("\nStarting Forward Selection Algorithm...\n");

	auto start = std::chrono::high_resolution_clock::now();
	
	// start with all features
	std::vector<int> remaining_features;
	remaining_features.reserve(16);
	for (int i = 1; i <= dataset.num_features_; i++) remaining_features.push_back(i);
	
	// tracks features selected through the search
	std::vector<int> selected_features;
	selected_features.reserve(16);
	
	// tracks overall best
	float overall_best_acc = default_rate(dataset);
	std::vector<int> overall_best_features;

	int depth = 0;

	fmt::print("\nSEARCHING DEPTH {}\n", depth);
	fmt::print("\tAcc {:.2f}% using feature(s): {} (DEFAULT RATE)\n", overall_best_acc, selected_features);
	fmt::print("DEPTH {} | BEST ACC: {:.2f}% | BEST FEATURES: {}\n", depth, overall_best_acc,
			selected_features);

	while (!remaining_features.empty()) {
		depth += 1;

		int level_best_feature = -1;
		float level_best_acc = -1.0;
		
		// loop through remaining features
		
		fmt::print("\nSEARCHING DEPTH {}\n", depth);
		for (const int& feature : remaining_features) {
			selected_features.push_back(feature);

			float acc = leave_one_out_validation(dataset.data_, selected_features);

			if (acc > level_best_acc) {
				level_best_acc = acc;
				level_best_feature = feature;
			}

			fmt::print("\tAcc {:.2f}% using feature(s): {} (+{})\n", acc,
					selected_features, feature);

			selected_features.pop_back();
		}

		// keep the best feature this depth
		selected_features.push_back(level_best_feature);
		std::erase(remaining_features, level_best_feature);

		fmt::print("DEPTH {} | BEST ACC: {:.2f}% | BEST FEATURES: {} (+{})\n", depth, level_best_acc, selected_features, level_best_feature);
		
		// record if this is the best found overall
		if (level_best_acc > overall_best_acc) {
			overall_best_acc = level_best_acc;
			overall_best_features = selected_features;
		}
	}

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> duration = end - start;
	
	fmt::print("\nForward Selection complete!!\n");
	fmt::print("Overall best feature set is {} with an accuracy of {:.2f}%\n",
			overall_best_features, overall_best_acc);
	fmt::print("Forward Selection Search Time: {:.2f}s\n", duration.count());
}

void backward_elimination(const Dataset& dataset) {
	fmt::print("\nStarting Backward Elimination Algorithm...\n");

	auto start = std::chrono::high_resolution_clock::now();
	
	// start with all features chosen
	std::vector<int> all_features;
	for (int i = 1; i <= dataset.num_features_; i++) all_features.push_back(i);

	std::vector<int> selected_features = all_features;
	
	// tracks overall best
	float overall_best_acc = leave_one_out_validation(dataset.data_, selected_features);
	std::vector<int> overall_best_features = selected_features;

	int depth = 0;

	fmt::print("\nSEARCHING DEPTH {}\n", depth);
	fmt::print("\tAcc {:.2f}% using feature(s): {}\n", overall_best_acc, selected_features);
	fmt::print("DEPTH {} | BEST ACC: {:.2f}% | BEST FEATURES: {}\n", depth, overall_best_acc,
			selected_features);

	while (selected_features.size() > 1) {
		depth += 1;

		int level_best_feature = -1;
		float level_best_acc = -1.0;
		
		// loop through remaining features
		
		fmt::print("\nSEARCHING DEPTH {}\n", depth);
		for (const int& feature : all_features) {
			std::erase(selected_features, feature);

			float acc = leave_one_out_validation(dataset.data_, selected_features);

			if (acc > level_best_acc) {
				level_best_acc = acc;
				level_best_feature = feature;
			}

			fmt::print("\tAcc {:.2f}% using feature(s): {} (-{})\n", acc, selected_features, feature);

			selected_features = all_features;
		}

		// erase feature that gave best acc this depth
		std::erase(all_features, level_best_feature);
		selected_features = all_features;

		fmt::print("DEPTH {} | BEST ACC: {:.2f}% | BEST FEATURES: {} (-{})\n", depth, level_best_acc,
				selected_features, level_best_feature);
		
		// record if this is the best found overall
		if (level_best_acc > overall_best_acc) {
			overall_best_acc = level_best_acc;
			overall_best_features = selected_features;
		}
	}
	
	depth += 1;
	float default_acc = default_rate(dataset);
	selected_features.clear();

	fmt::print("\nSEARCHING DEPTH {}\n", depth);
	fmt::print("\tAcc {:.2f}% using feature(s): {} (DEFAULT RATE)\n", default_acc, selected_features);
	fmt::print("DEPTH {} | BEST ACC: {:.2f}% | BEST FEATURES: {}\n", depth, default_acc,
			selected_features);


	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float> duration = end - start;
	
	fmt::print("\nForward Selection complete!!\n");
	fmt::print("Overall best feature set is {} with an accuracy of {:.2f}%\n",
			overall_best_features, overall_best_acc);
	fmt::print("Forward Selection Search Time: {:.2f}s\n", duration.count());
}


#endif
