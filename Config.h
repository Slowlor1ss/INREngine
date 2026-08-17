#pragma once
#include <string>
#include <vector>

#include "ImageGenerator.h"
#include "Network.h"
#include "WindowRenderer.h"
#include "Types.h"

namespace config
{
	inline bool use_gpu = true;
	inline bool benchmark_enabled = false;

	inline std::string target_image_file = "Training_Data/0064_x4.bmp"; //"Training_Data/DIV2K_train_LR_mild/0064x4m.bmp";// "Training_Data/0064_x4.bmp";
	inline std::string output_path = "";
	inline std::string output_filename = target_image_file;

	inline engineFloat output_image_scale = 4.f;
	inline std::vector<size_t> custom_layer_dims = { 256, 256, 3 };
	inline std::vector<ActFunc::Base*> custom_activations = {
		ActFunc::DataBase::FindActFunc<ActFunc::Finer>(),
		ActFunc::DataBase::FindActFunc<ActFunc::Finer>(),
		ActFunc::DataBase::FindActFunc<ActFunc::None>()
	};

	inline RenderMode render_mode = RenderMode::StandardRGB;

	inline bool use_grid_encoding = true;
	inline bool use_positional_encoding = false;
	inline int pe_num_frequencies = 10; // Positional encode
	inline bool use_gaussian_pe = false;

	inline bool initial_live_update_state = true;

	// Hyperparameters & Training State
	// Note if we drop this below out thread count we will run singlethreaded (which should be fine)
	inline size_t batch_size = 256ull * 256ull;//510 * 338; // 256ull*256ull;//8192;//65536;//8192;//32;
	inline bool shuffle_pixel_batch = true; // TODO: either make this an input parameter or make this the default if batch size isnt == to image size
	inline size_t print_every_n_batches = 1000;
	//inline float initial_learning_rate = 0.0001f;
	inline engineFloat initial_learning_rate = 0.1f;//0.005f;//0.005f;//WIRE //0.000025f; Siren
	inline size_t max_epochs = 40'000; // From the PyTorch script niters = 2000

	// Hyperparameter to balance how much the network cares about slopes vs colors
	// used by spatial gradient (use 0 to turn spatial gradient off)
	//inline engineFloat spatialLossWeight = 0.f; // TODO: make a utils file so we can use this
}
