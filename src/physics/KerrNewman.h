#pragma once
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <cmath>

// Kerr-Newman black hole parameters (geometric units: G=c=1)
struct BHParams {
    float M = 1.0f;   // mass
    float a = 0.0f;   // spin (a = J/M, |a| <= M)
    float Q = 0.0f;   // charge (|Q| <= sqrt(M^2 - a^2))
};

// Boyer-Lindquist coordinates state for photon geodesic
struct GeodesicState {
    float r, theta, phi, t;
    float dr, dtheta, dphi, dt; // derivatives w.r.t. affine parameter
};

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace KN {

CUDA_HOSTDEV inline float sigma(float r, float theta, float a) {
    float costh = cosf(theta);
    return r * r + a * a * costh * costh;
}

CUDA_HOSTDEV inline float delta(float r, float M, float a, float Q) {
    return r * r - 2.0f * M * r + a * a + Q * Q;
}

// Photon geodesic equations in Kerr-Newman metric
// Returns (dr/dlambda, dtheta/dlambda) signs from potentials R(r) and Theta(theta)
// E=energy, L=azimuthal angular momentum, K=Carter constant (normalized: E=1)
CUDA_HOSTDEV inline bool computeGeodesicRHS(
    float r, float theta,
    float M, float a, float Q,
    float b,   // impact parameter b = L/E
    float q2,  // q^2 = K/E^2  (Carter constant / E^2)
    float& R_r, float& Theta_th)
{
    float sin_th = sinf(theta);
    float cos_th = cosf(theta);
    float sig = r * r + a * a * cos_th * cos_th;
    float del = r * r - 2.0f * M * r + a * a + Q * Q;

    // R(r) = [(r^2+a^2) - a*b]^2 - delta*[q^2 + (b-a)^2]
    float tmp = (r * r + a * a) - a * b;
    R_r = tmp * tmp - del * (q2 + (b - a) * (b - a));

    // Theta(theta) = q^2 + a^2*cos^2(theta) - b^2/sin^2(theta) + a^2 (no, standard form:)
    // Theta(theta) = q^2 - cos^2(theta)*(b^2/sin^2(theta) - a^2)
    float cot2 = cos_th * cos_th;
    float sin2 = sin_th * sin_th;
    if (sin2 < 1e-8f) sin2 = 1e-8f;
    Theta_th = q2 - cot2 * (b * b / sin2 - a * a);

    return (R_r >= 0.0f && Theta_th >= 0.0f);
}

// d(phi)/d(lambda) * Sigma
CUDA_HOSTDEV inline float dphidlam_sigma(float r, float theta,
    float M, float a, float Q, float b)
{
    float sin_th = sinf(theta);
    float sin2 = sin_th * sin_th;
    if (sin2 < 1e-8f) sin2 = 1e-8f;
    float del = r * r - 2.0f * M * r + a * a + Q * Q;
    // dphi/dlambda * Sigma = -(a - b/sin^2) + a/Delta * [(r^2+a^2) - a*b]
    return -(a - b / sin2) + a / del * ((r * r + a * a) - a * b);
}

// d(t)/d(lambda) * Sigma
CUDA_HOSTDEV inline float dtdlam_sigma(float r, float theta,
    float M, float a, float Q, float b)
{
    float sin_th = sinf(theta);
    float sin2 = sin_th * sin_th;
    if (sin2 < 1e-8f) sin2 = 1e-8f;
    float del = r * r - 2.0f * M * r + a * a + Q * Q;
    // dt/dlambda * Sigma = -a*(a*sin^2 - b) + (r^2+a^2)/Delta * [(r^2+a^2) - a*b]
    return -a * (a * sin2 - b) + (r * r + a * a) / del * ((r * r + a * a) - a * b);
}

// Horizon radius (fmaxf guards against naked-singularity inputs producing NaN)
CUDA_HOSTDEV inline float r_plus(float M, float a, float Q) {
    return M + sqrtf(fmaxf(M * M - a * a - Q * Q, 0.0f));
}

// Innermost stable circular orbit (ISCO) for Kerr
CUDA_HOSTDEV inline float r_isco(float M, float a) {
    // Bardeen et al. 1972 formula
    float z1 = 1.0f + cbrtf(1.0f - a*a/(M*M)) *
               (cbrtf(1.0f + a/M) + cbrtf(1.0f - a/M));
    float z2 = sqrtf(3.0f * a*a/(M*M) + z1*z1);
    float sign = (a >= 0) ? 1.0f : -1.0f;
    return M * (3.0f + z2 - sign * sqrtf((3.0f - z1)*(3.0f + z1 + 2.0f*z2)));
}

} // namespace KN
