// STDLIB
#include <iostream>
#include <string>

// ORIGINAL HEADERS

#include "data.h"
#include "nn.h"
#include "selection.h"

// MAIN PROGRAM CODE

int main() {
	fmt::print("Welcome to the Feature Selection Program!\n");
	const Dataset dataset = read_dataset();

	fmt::print("\n== DATASET STATS ==\n");
	fmt::print("TOTAL ROWS: {} | FEATURES: {} | LABEL ONE COUNT: {} | LABEL TWO COUNT: {}\n",
			dataset.data_.size(), dataset.num_features_, 
			dataset.label_one_count_, dataset.label_two_count_);

	fmt::print("\nPlease choose a feature selection algorithm.\n\t1) Forward Selection\n\t2) Backward Elimination\n");


	while (true) {
		fmt::print("(1/2): ");
		std::string algo_choice = "";
		std::cin >> algo_choice;

		if (algo_choice == "1") {
			forward_selection(dataset);
			break;
		}
		if (algo_choice == "2") {
			backward_elimination(dataset);
			break;
		}
		fmt::print("ERROR: Invalid algorithm choice. Please enter 1 or 2.\n");
	}

	return 0;
}
