#pragma once
#include <cuda_runtime.h>
#include <string>
#include <vector>

struct CudaTexture {
    cudaTextureObject_t tex = 0;
    cudaArray_t         arr = nullptr;
    int width = 0, height = 0;
    // Host-side pixel storage for CPU renderer (populated when useCuda=false).
    std::vector<float4> cpu_pixels;
    bool valid() const { return tex != 0 || !cpu_pixels.empty(); }
    void free();
    // Bilinear sample: u wraps [0,1], v clamps [0,1]. Used by CPU renderer.
    float4 sample(float u, float v) const;
};

// Call before creating any textures. Skips GPU upload and populates cpu_pixels
// instead. Must be called in BlackHoleWidget constructor before initializeGL.
void setUseCuda(bool v);

// Load any image format supported by Qt (JPG, PNG, BMP, EXR if plugin present)
CudaTexture loadTexture(const std::string& path);
CudaTexture makeSolidTexture(float r, float g, float b, float a = 1.0f);
// Procedural star field (equirectangular, 2048x1024 by default)
CudaTexture makeStarfieldTexture(int w = 2048, int h = 1024);
