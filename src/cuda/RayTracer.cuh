#pragma once
#include <cuda_runtime.h>
#include "physics/KerrNewman.h"

struct RenderParams {
    int   width, height;
    float fov;            // horizontal field of view in radians
    // Camera in Cartesian-like coords (converted to BL internally)
    float cam_r;          // camera distance from BH
    float cam_theta;      // polar angle
    float cam_phi;        // azimuthal angle
    float cam_up_phi;     // rotation of "up" vector
    float cam_look_yaw;   // look direction: horizontal offset from BH-pointing baseline
    float cam_look_pitch; // look direction: vertical offset
    BHParams bh;
    bool  show_disk;
    float disk_inner;     // inner disk radius
    float disk_outer;     // outer disk radius
    int   max_steps;      // geodesic integration steps
    float step_size;      // affine parameter step
    float jitter_x;       // sub-pixel jitter in NDC units (Halton sequence for TAA)
    float jitter_y;
    float sky_lon_offset; // equirectangular sky longitude offset [0,1): rotates seam position
    int   debug_mode;     // 0=normal, 1=debug colors by termination status
};

// Launch the CUDA ray tracer kernel
void launchRayTracer(
    float4*       d_output,
    cudaTextureObject_t bg_tex,
    cudaTextureObject_t disk_tex,
    const RenderParams& params,
    cudaStream_t  stream = nullptr
);

// Blend newFrame into accum: accum = accum*(1-alpha) + newFrame*alpha
void launchTemporalBlend(float4* accum, const float4* newFrame,
                          int w, int h, float alpha,
                          cudaStream_t stream = nullptr);
