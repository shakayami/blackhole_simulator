#pragma once
#include "TextureLoader.h"
#include "cuda/RayTracer.cuh"

// CPU-parallel fallback ray tracer using std::thread.
// Mirrors the GPU kernel logic exactly; used when no CUDA device is present.
void launchRayTracerCPU(float4*             output,
                         const CudaTexture& bg_tex,
                         const CudaTexture& disk_tex,
                         const RenderParams& params);

void launchTemporalBlendCPU(float4*       accum,
                             const float4* src,
                             int w, int h, float alpha);
