#ifndef SELECTION_H
#define SELECTION_H

#include <chrono>
#include <set>

#include <fmt/base.h>
#include <fmt/ranges.h>

#include "data.h"
#include "nn.h"

void forward_selection(const Dataset& dataset) {
	fmt::print("\nStarting Forward Selection Algorithm...\n");

	auto start = std::chrono::high_resolution_clock::now();
	
	// start with all features
	std::set<int> remaining_features;
	for (int i = 1; i <= dataset.num_features_; i++) remaining_features.insert(i);
	
	// tracks features selected through the search
	std::set<int> selected_features;
	
	// tracks overall best
	float overall_best_acc = default_rate(dataset);
	std::set<int> overall_best_features;

	int depth = 0;

	fmt::print("\nSEARCHING DEPTH {}\n", depth);
	fmt::print("\tAcc {:.2f}% using feature(s): {} (DEFAULT RATE)\n", overall_best_acc, selected_features);
	fmt::print("DEPTH {} | BEST ACC: {:.2f}% | BEST FEATURES: {}\n", depth, overall_best_acc,
			selected_features);

	while (!remaining_features.empty()) {
		depth += 1;

		int level_best_feature = -1;
		float level_best_acc = -1.0f;
		
		// loop through remaining features
		
		fmt::print("\nSEARCHING DEPTH {}\n", depth);
		for (const int& feature : remaining_features) {
			selected_features.insert(feature);

			float acc = leave_one_out_validation(dataset.data_, selected_features);

			if (acc > level_best_acc) {
				level_best_acc = acc;
				level_best_feature = feature;
			}

			fmt::print("\tAcc {:.2f}% using feature(s): {}\n", acc, selected_features);

			selected_features.erase(feature);
		}

		// keep the best feature this depth
		selected_features.insert(level_best_feature);
		remaining_features.erase(level_best_feature);

		fmt::print("DEPTH {} | BEST ACC: {:.2f}% | BEST FEATURES: {}\n", depth, level_best_acc,
				selected_features);
		
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

#endif
