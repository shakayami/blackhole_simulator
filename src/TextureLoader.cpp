#define _USE_MATH_DEFINES
#include "TextureLoader.h"
#include <stb_image.h>
#include <QImage>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>

static bool s_useCuda = true;
void setUseCuda(bool v) { s_useCuda = v; }

// ── Core upload helper ────────────────────────────────────────────────────────
// If s_useCuda: create CUDA texture object.
// Otherwise: store pixels in cpu_pixels for the CPU renderer.

static CudaTexture buildTexture(const float* rgba, int w, int h) {
    CudaTexture result;
    result.width  = w;
    result.height = h;

    if (s_useCuda) {
        cudaChannelFormatDesc desc = cudaCreateChannelDesc<float4>();
        cudaMallocArray(&result.arr, &desc, w, h);
        cudaMemcpy2DToArray(result.arr, 0, 0, rgba,
                            w * 4 * sizeof(float),
                            w * 4 * sizeof(float), h,
                            cudaMemcpyHostToDevice);

        cudaResourceDesc resDesc{};
        resDesc.resType         = cudaResourceTypeArray;
        resDesc.res.array.array = result.arr;

        cudaTextureDesc texDesc{};
        texDesc.addressMode[0]   = cudaAddressModeWrap;
        texDesc.addressMode[1]   = cudaAddressModeClamp;
        texDesc.filterMode       = cudaFilterModeLinear;
        texDesc.readMode         = cudaReadModeElementType;
        texDesc.normalizedCoords = 1;

        cudaCreateTextureObject(&result.tex, &resDesc, &texDesc, nullptr);
    } else {
        result.cpu_pixels.resize(w * h);
        for (int i = 0; i < w * h; ++i)
            result.cpu_pixels[i] = {rgba[i*4+0], rgba[i*4+1], rgba[i*4+2], rgba[i*4+3]};
    }
    return result;
}

void CudaTexture::free() {
    if (tex) { cudaDestroyTextureObject(tex); tex = 0; }
    if (arr) { cudaFreeArray(arr); arr = nullptr; }
    cpu_pixels.clear();
    cpu_pixels.shrink_to_fit();
}

// Bilinear sample with wrap-u / clamp-v.
float4 CudaTexture::sample(float u, float v) const {
    if (cpu_pixels.empty()) return {0.0f, 0.0f, 0.0f, 1.0f};
    u = u - std::floor(u);                           // wrap u to [0,1)
    v = std::max(0.0f, std::min(1.0f, v));           // clamp v to [0,1]

    float fx = u * (float)width  - 0.5f;
    float fy = v * (float)height - 0.5f;
    int x0 = (int)std::floor(fx);
    int y0 = (int)std::floor(fy);
    float tx = fx - (float)x0;
    float ty = fy - (float)y0;

    auto fetch = [&](int x, int y) -> float4 {
        x = ((x % width)  + width)  % width;
        y = std::max(0, std::min(height - 1, y));
        return cpu_pixels[y * width + x];
    };
    float4 c00 = fetch(x0,   y0);
    float4 c10 = fetch(x0+1, y0);
    float4 c01 = fetch(x0,   y0+1);
    float4 c11 = fetch(x0+1, y0+1);
    auto lerp4 = [](float4 a, float4 b, float t) -> float4 {
        float s = 1.0f - t;
        return {a.x*s + b.x*t, a.y*s + b.y*t, a.z*s + b.z*t, a.w*s + b.w*t};
    };
    return lerp4(lerp4(c00, c10, tx), lerp4(c01, c11, tx), ty);
}

// ── Public texture constructors ───────────────────────────────────────────────

CudaTexture loadTexture(const std::string& path) {
    int w = 0, h = 0, channels = 0;
    stbi_set_flip_vertically_on_load(0);
    float* data = stbi_loadf(path.c_str(), &w, &h, &channels, 4);

    if (!data) {
        QImage img(QString::fromStdString(path));
        if (img.isNull())
            return makeSolidTexture(0.05f, 0.05f, 0.15f);
        img = img.convertToFormat(QImage::Format_RGBA8888);
        w = img.width(); h = img.height();
        std::vector<float> rgba(w * h * 4);
        const uchar* bits = img.bits();
        for (int i = 0; i < w * h; ++i) {
            float r = bits[i*4+0] / 255.0f, g = bits[i*4+1] / 255.0f;
            float b = bits[i*4+2] / 255.0f, a = bits[i*4+3] / 255.0f;
            rgba[i*4+0] = r*r; rgba[i*4+1] = g*g;
            rgba[i*4+2] = b*b; rgba[i*4+3] = a;
        }
        return buildTexture(rgba.data(), w, h);
    }
    CudaTexture result = buildTexture(data, w, h);
    stbi_image_free(data);
    return result;
}

CudaTexture makeSolidTexture(float r, float g, float b, float a) {
    float rgba[4] = {r, g, b, a};
    return buildTexture(rgba, 1, 1);
}

// ── Procedural star field ─────────────────────────────────────────────────────

static inline uint32_t pcg32(uint32_t x) {
    x = x * 747796405u + 2891336453u;
    x = ((x >> ((x >> 28) + 4)) ^ x) * 277803737u;
    return (x >> 22) ^ x;
}
static inline float randf(uint32_t& seed) {
    seed = pcg32(seed);
    return (seed >> 8) / float(1 << 24);
}

CudaTexture makeStarfieldTexture(int w, int h) {
    std::vector<float> rgba(w * h * 4, 0.0f);

    for (int i = 0; i < w * h; ++i) rgba[i*4+3] = 1.0f;

    // ── Milky Way band ────────────────────────────────────────────────────────
    for (int y = 0; y < h; ++y) {
        float lat = (0.5f - (y + 0.5f) / h) * (float)M_PI;
        float mw_base = 0.06f * std::exp(-lat * lat / 0.18f);
        for (int x = 0; x < w; ++x) {
            float phi = (x + 0.5f) / w * 2.0f * (float)M_PI;
            float arm  = 0.5f + 0.3f * std::sin(phi * 2.3f + 0.7f)
                               + 0.2f * std::cos(phi * 5.1f - 1.2f);
            uint32_t seed = (uint32_t)(y * w + x) * 2246822519u;
            float noise = 0.4f + 0.6f * randf(seed);
            float intensity = mw_base * arm * noise;
            int idx = (y * w + x) * 4;
            float edge = 1.0f - std::abs(lat) / 0.8f;
            rgba[idx+0] += intensity * (0.65f + 0.25f * edge);
            rgba[idx+1] += intensity * (0.75f + 0.15f * edge);
            rgba[idx+2] += intensity;
        }
    }

    // ── Stars ─────────────────────────────────────────────────────────────────
    auto splatStar = [&](int cx, int cy, float sr, float sg, float sb,
                          float brightness, int radius) {
        for (int dy = -radius; dy <= radius; ++dy) {
            int iy = cy + dy;
            if (iy < 0 || iy >= h) continue;
            for (int dx = -radius; dx <= radius; ++dx) {
                float d2 = float(dx*dx + dy*dy);
                float sigma = radius > 1 ? float(radius) * 0.7f : 0.8f;
                float w_gauss = std::exp(-d2 / (2.0f * sigma * sigma));
                int ix = ((cx + dx) % w + w) % w;
                int idx = (iy * w + ix) * 4;
                rgba[idx+0] = std::min(1.0f, rgba[idx+0] + sr * brightness * w_gauss);
                rgba[idx+1] = std::min(1.0f, rgba[idx+1] + sg * brightness * w_gauss);
                rgba[idx+2] = std::min(1.0f, rgba[idx+2] + sb * brightness * w_gauss);
            }
        }
    };

    // Faint background stars
    {
        const int cell = 6;
        int nx = w / cell, ny = h / cell;
        for (int cy = 0; cy < ny; ++cy) {
            for (int cx = 0; cx < nx; ++cx) {
                uint32_t seed = (uint32_t)(cy * nx + cx) * 1664525u + 1013904223u;
                if (randf(seed) > 0.42f) continue;
                float fx = randf(seed), fy = randf(seed);
                int px = cx * cell + int(fx * cell);
                int py = cy * cell + int(fy * cell);
                float b = std::pow(randf(seed), 2.5f) * 0.35f + 0.04f;
                float lat = (0.5f - (py + 0.5f) / h) * (float)M_PI;
                b *= 1.0f + 0.6f * std::exp(-lat * lat / 0.25f);
                float r32 = randf(seed);
                float sr = 1.0f, sg = 1.0f, sb = 1.0f;
                if      (r32 < 0.15f) { sr = 0.80f; sg = 0.88f; sb = 1.00f; }
                else if (r32 < 0.40f) { sr = 0.95f; sg = 0.97f; sb = 1.00f; }
                else if (r32 < 0.65f) { sr = 1.00f; sg = 0.97f; sb = 0.80f; }
                else                  { sr = 1.00f; sg = 0.78f; sb = 0.55f; }
                splatStar(px, py, sr, sg, sb, b, 1);
            }
        }
    }

    // Bright named-star layer
    {
        const int cell = 48;
        int nx = w / cell, ny = h / cell;
        for (int cy = 0; cy < ny; ++cy) {
            for (int cx = 0; cx < nx; ++cx) {
                uint32_t seed = (uint32_t)(cy * nx + cx + 99991u) * 1664525u + 1013904223u;
                if (randf(seed) > 0.28f) continue;
                float fx = randf(seed), fy = randf(seed);
                int px = cx * cell + int(fx * cell);
                int py = cy * cell + int(fy * cell);
                float b = 0.5f + randf(seed) * 0.5f;
                float r32 = randf(seed);
                float sr = 1.0f, sg = 1.0f, sb = 1.0f;
                if      (r32 < 0.20f) { sr = 0.70f; sg = 0.82f; sb = 1.00f; }
                else if (r32 < 0.50f) { sr = 1.00f; sg = 1.00f; sb = 0.90f; }
                else if (r32 < 0.75f) { sr = 1.00f; sg = 0.90f; sb = 0.60f; }
                else                  { sr = 1.00f; sg = 0.55f; sb = 0.35f; }
                splatStar(px, py, sr, sg, sb, b,       1);
                splatStar(px, py, sr, sg, sb, b * 0.3f, 3);
            }
        }
    }

    return buildTexture(rgba.data(), w, h);
}
