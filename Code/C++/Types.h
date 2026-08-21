#pragma once

// This allows us to easily swap our float type to e.g. Bfloat or a 16bit float or even double
// TODO: we need to do a lot more casts all throughout our engine as we really decided to add this way to late
// NOTE: running with double on the gpu is probably going to be a lot slower, im unsure how bfloats would be handled so TODO: look in to this
using engineFloat = float;

//#include <cuda_bf16.h>
//using engineFloat = __nv_bfloat16;

//#include <cuda_fp16.h>
//using engineFloat = __half;

