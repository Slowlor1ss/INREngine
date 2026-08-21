#pragma once
#include <cstdint>

// !!!
// NOTE: Must stay in sync with python/viz/shared_memory_bridge.py 
// same field order, same widths, packed with no padding 
// !!!
#pragma pack(push, 1)
struct VizSharedHeader
{
    uint32_t magic = 0x56524E49; // 'INRV', sanity check so Python doesn't attach to garbage (INRV stands for Implicit Neural Representation Viewer which is what this project is and the py is the viewer... get it, very meta, haha... okey, youre not actually supposed ot read this far... ahum... right... I need sleep)
    uint32_t _PLACEHOLDER = 1;  // Can be replaced I just needed one more 32bit value :)
    uint32_t frameCounter = 0;  // Seqlock: even = stable, odd = write in progress
    uint32_t width = 0;         // Dimensions of the "Current" frame buffer
    uint32_t height = 0;        //
    uint32_t channels = 0;      // 3 for RGB etc...
    uint32_t hasTarget = 0;     // 1 if the "Target/Reference" mapping was created
    uint32_t targetWidth = 0;   // Dimensions of our INR generated frame buffer
    uint32_t targetHeight = 0;  //
    uint32_t running = 0;       // 1 while the training process is alive 0 if not... (changed in ~PythonVisualizerBridge)
                                // this way we can tell "no new frames" apart from "we quit"
    uint64_t epoch = 0;
    float cost = 0.0f;
    float learningRate = 0.0f;
};
#pragma pack(pop)


// Hopefully this'll prevents some future bugs *sigh*
static_assert(sizeof(VizSharedHeader) == 56, "VizSharedHeader layout changed update python/viz/shared_memory_bridge.py to match");
