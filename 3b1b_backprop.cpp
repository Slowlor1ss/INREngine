#include "helpers.h"
#include "NeuralImageRecreator.hpp"
#include "MNISTAutoencoder.hpp"

// TODO: don't see the point for the database, just use them directly? to have a single instance? does that matter?
// seems to be to use the inheritance more easily >> convert to template instead?
// 
// TODO: Relu needs HE initialization: make it so that happens automatically when selecting relu, and not when selecting sigmoid for a layer.
// TODO: serialization
// TODO: better interface / frontend.

// ============================================================================
// Main Execution
// ============================================================================

int main(int argc, char** argv)
{
	// 1. Seed Random Number Generator
	auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	srand(static_cast<uint32_t>(seed));

	// Parse the command line arguments right at startup
	ParseCommandLine(argc, argv);

	// Setup 1: Neural Image Recreator (Uncomment to use)
	NeuralImageRecreator();

	// Setup 2: MNIST Autoencoder & Latent Space Explorer
	RunMNISTAutoencoder();

	return 0;
}
