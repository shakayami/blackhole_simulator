#include "RayTracer.cuh"
#include "physics/KerrNewman.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

// ── Vector helpers ────────────────────────────────────────────────────────────

__device__ inline float dot3(float ax, float ay, float az,
                              float bx, float by, float bz) {
    return ax*bx + ay*by + az*bz;
}
__device__ inline void cross3(float ax, float ay, float az,
                               float bx, float by, float bz,
                               float& rx, float& ry, float& rz) {
    rx = ay*bz - az*by; ry = az*bx - ax*bz; rz = ax*by - ay*bx;
}
__device__ inline void normalize3(float& x, float& y, float& z) {
    float inv = rsqrtf(x*x + y*y + z*z); x *= inv; y *= inv; z *= inv;
}
// fallback_phi: accumulated BL φ at escape (normalized to [-π,π]).
// Used when the exit direction is near the celestial poles (r_xy < 1e-6) where
// atan2(dy,dx) is numerically undefined.  This gives continuity with adjacent
// pixels instead of a fixed 0.5 that creates a pole-meridian step artifact.
__device__ inline void dirToUV(float dx, float dy, float dz, float& u, float& v,
                                 float fallback_phi = 0.0f) {
    float inv = rsqrtf(dx*dx + dy*dy + dz*dz);
    dx *= inv; dy *= inv; dz *= inv;
    float cz = fmaxf(-1.0f, fminf(1.0f, dz));
    v = acosf(cz) / CUDART_PI_F;
    float r_xy = sqrtf(fmaxf(1.0f - cz*cz, 0.0f));
    if (r_xy > 1e-6f) {
        u = atan2f(dy, dx) / (2.0f * CUDART_PI_F) + 0.5f;
    } else {
        // Near poles: derive u from the ray's escape φ for neighbor-continuity.
        u = fallback_phi / (2.0f * CUDART_PI_F) + 0.5f;
        u = u - floorf(u);
    }
}


// Procedural lat/lon grid background (debug_mode 5).
// u in [0,1): longitude as circular variable on S^1.  u=0 ≡ u=1.
// v in [0,1]:  latitude (v=0 = north pole, v=1 = south pole).
//
// Longitude lines use circular distance min(frac, 1-frac) which is continuous
// and periodic on S^1 — the grid has no seam by construction.
__device__ inline float4 proceduralGrid(float u, float v) {
    constexpr int   N_lon  = 24;      // 24 meridians  (every 15°)
    constexpr int   N_lat  = 12;      // 12 parallels  (every 15°)
    constexpr float half_w = 0.040f;  // half line-width as fraction of cell

    float lf    = (u - floorf(u)) * (float)N_lon;
    float lon_f = lf - floorf(lf);                     // [0,1) within cell
    float lon_d = fminf(lon_f, 1.0f - lon_f);          // circular dist [0,0.5]

    float laf   = v * (float)N_lat;
    float lat_f = laf - floorf(laf);
    float lat_d = fminf(lat_f, 1.0f - lat_f);

    if (lon_d < half_w || lat_d < half_w)
        return {0.80f, 0.85f, 1.00f, 1.0f};  // blue-white grid line
    return {0.04f, 0.06f, 0.12f, 1.0f};       // dark navy background
}

// ── Disk ─────────────────────────────────────────────────────────────────────

__device__ inline float diskNoise(float r, float phi) {
    return 0.78f
         + 0.13f * sinf(r * 2.3f + phi * 0.9f)
         + 0.09f * sinf(r * 4.7f - phi * 2.1f);
}

__device__ inline float4 diskColor(float r, float phi, float cam_phi,
                                    float disk_inner, float disk_outer,
                                    float M, float a) {
    if (r < disk_inner || r > disk_outer) return {0,0,0,0};
    float t    = (r - disk_inner) / fmaxf(disk_outer - disk_inner, 1e-6f);
    float temp = powf(fmaxf(1.0f - t, 0.005f), 0.75f);
    // Smooth C-inf color ramp (no derivative kinks that alias in the lensed image)
    float R = 0.6f + 0.4f * temp;
    float G = temp * (2.0f - temp);       // smooth 0→1 over [0,1]
    float B = temp * temp * temp;
    float v     = fminf(sqrtf(M / fmaxf(r, M * 1.05f)), 0.5f);
    float dop   = fmaxf(0.4f, 1.0f + v * sinf(cam_phi - phi));
    float boost = dop * dop;
    // Novikov-Thorne: emission → 0 at ISCO (physically correct inner BC).
    // Without this the inner edge is the brightest point; combined with strong
    // lensing near the photon sphere for spinning BH this creates a wide bright band.
    float nt    = 1.0f - sqrtf(disk_inner / fmaxf(r, disk_inner * 1.001f));
    float lum   = temp * boost * diskNoise(r, phi) * nt;

    float fade_i = 1.0f - expf(-(r - disk_inner) / fmaxf(disk_inner * 0.08f, 0.4f));
    float fade_o = 1.0f - expf(-(disk_outer - r) / fmaxf(disk_outer * 0.06f, 0.8f));
    float opacity = 0.80f * fade_i * fade_o;
    return {R*lum, G*lum, B*lum, opacity};
}

// ── Kerr-Newman geodesic (Mino time) ─────────────────────────────────────────

struct KNState { float r, theta, phi; };

__device__ inline KNState kn_deriv(KNState s, float b, float q2,
                                    float sr, float sth,
                                    float M, float a, float Qbh) {
    float Rr, Tth;
    KN::computeGeodesicRHS(s.r, s.theta, M, a, Qbh, b, q2, Rr, Tth);
    return { sr  * sqrtf(fmaxf(Rr,  0.0f)),
             sth * sqrtf(fmaxf(Tth, 0.0f)),
             KN::dphidlam_sigma(s.r, s.theta, M, a, Qbh, b) };
}

// Bisect to find the root of Theta(theta)=0 between th_in (Theta>=0) and th_out (Theta<0).
// Works regardless of whether th_in < th_out or th_in > th_out.
__device__ inline float thetaTurnBisect(float th_in, float th_out,
                                         float b, float q2, float a) {
    for (int i = 0; i < 12; ++i) {
        float th_mid = (th_in + th_out) * 0.5f;
        float c = cosf(th_mid), s2 = fmaxf(1.0f - c*c, 1e-8f);
        if (q2 - c*c * (b*b/s2 - a*a) >= 0.0f) th_in  = th_mid;
        else                                      th_out = th_mid;
    }
    return (th_in + th_out) * 0.5f;
}

// Bisect to find R(r)=0 between r_in (R>=0) and r_out (R<0).
// R(r) is independent of theta; any theta value works (1.0f used here).
__device__ inline float rTurnBisect(float r_in, float r_out,
                                     float b, float q2,
                                     float M, float a, float Qbh) {
    for (int i = 0; i < 12; ++i) {
        float r_mid = (r_in + r_out) * 0.5f;
        float Rm, dummy;
        KN::computeGeodesicRHS(r_mid, 1.0f, M, a, Qbh, b, q2, Rm, dummy);
        if (Rm >= 0.0f) r_in  = r_mid;
        else            r_out = r_mid;
    }
    return (r_in + r_out) * 0.5f;
}

__device__ KNState kn_rk4(KNState s, float b, float q2,
                           float& sr, float& sth,
                           float M, float a, float Qbh, float h) {
    KNState k1 = kn_deriv(s, b, q2, sr, sth, M, a, Qbh);
    KNState s2 = {s.r+0.5f*h*k1.r, s.theta+0.5f*h*k1.theta, s.phi+0.5f*h*k1.phi};
    KNState k2 = kn_deriv(s2, b, q2, sr, sth, M, a, Qbh);
    KNState s3 = {s.r+0.5f*h*k2.r, s.theta+0.5f*h*k2.theta, s.phi+0.5f*h*k2.phi};
    KNState k3 = kn_deriv(s3, b, q2, sr, sth, M, a, Qbh);
    KNState s4 = {s.r+h*k3.r, s.theta+h*k3.theta, s.phi+h*k3.phi};
    KNState k4 = kn_deriv(s4, b, q2, sr, sth, M, a, Qbh);
    KNState nx = {
        s.r     + h/6.0f*(k1.r     + 2*k2.r     + 2*k3.r     + k4.r),
        s.theta + h/6.0f*(k1.theta + 2*k2.theta + 2*k3.theta + k4.theta),
        s.phi   + h/6.0f*(k1.phi   + 2*k2.phi   + 2*k3.phi   + k4.phi)
    };
    float Rn, Tn;
    KN::computeGeodesicRHS(nx.r, nx.theta, M, a, Qbh, b, q2, Rn, Tn);
    KNState nx0 = nx;  // save original RK4 result before turning-point correction

    if (Rn < 0.0f) {
        // RK4 overshot the radial turning point.  Bisect to exact R(r)=0,
        // interpolate (theta, phi), flip sr, then advance analytically for the
        // remaining Mino interval — same approach as the theta turning point.
        float r_turn = rTurnBisect(s.r, nx0.r, b, q2, M, a, Qbh);
        float dr_total = nx0.r - s.r;
        float frac_r = (fabsf(dr_total) > 1e-10f)
                       ? fmaxf(0.0f, fminf(1.0f, (r_turn - s.r) / dr_total))
                       : 0.5f;
        KNState s_rturn = { r_turn,
                            s.theta + frac_r * (nx0.theta - s.theta),
                            s.phi   + frac_r * (nx0.phi   - s.phi) };
        float sr_before = sr;
        sr = -sr;
        float h_rem_r = h * (1.0f - frac_r);

        // theta, phi: standard Euler (derivatives non-zero at r_turn)
        KNState k1rt = kn_deriv(s_rturn, b, q2, sr, sth, M, a, Qbh);

        // r: analytic quadratic advance
        // A = |dR/dr| at r_turn, estimated toward the allowed (R>0) side
        float eps_fd = 1e-5f;
        float r_fd   = r_turn - sr_before * eps_fd;
        float R_fd, dummy_t;
        KN::computeGeodesicRHS(r_fd, s_rturn.theta, M, a, Qbh, b, q2, R_fd, dummy_t);
        float A_r    = fmaxf(R_fd, 0.0f) / eps_fd;
        float delta_r = (-sr_before) * A_r * h_rem_r * h_rem_r * 0.25f;

        nx = { r_turn        + delta_r,
               s_rturn.theta + h_rem_r * k1rt.theta,
               s_rturn.phi   + h_rem_r * k1rt.phi };
    }

    if (Tn < 0.0f) {
        // RK4 overshot the theta turning point.  Bisect to the exact Theta=0
        // crossing, interpolate (r,phi), flip sth, then continue for the
        // remaining Mino interval.
        //
        // Use nx0 (original RK4 result) for bisection bounds so a simultaneous
        // r-turn does not corrupt the theta-turn fraction computation.
        //
        // Near the turning point Theta ≈ A*(theta - theta_turn), so the exact
        // solution is theta(delta_lambda) = theta_turn + sth_new * A * delta_lambda^2 / 4.
        float theta_turn = thetaTurnBisect(s.theta, nx0.theta, b, q2, a);
        float dth_total  = nx0.theta - s.theta;
        float frac = (fabsf(dth_total) > 1e-10f)
                     ? fmaxf(0.0f, fminf(1.0f, (theta_turn - s.theta) / dth_total))
                     : 0.5f;
        KNState s_turn = { s.r   + frac * (nx0.r   - s.r),
                           theta_turn,
                           s.phi + frac * (nx0.phi - s.phi) };
        float sth_before = sth;
        sth = -sth;
        float h_rem = h * (1.0f - frac);

        // r, phi: standard Euler (derivatives non-zero at theta_turn)
        KNState k1r = kn_deriv(s_turn, b, q2, sr, sth, M, a, Qbh);

        // theta: analytic quadratic advance
        // A = |dTheta/dtheta| at theta_turn, estimated toward the allowed (Theta>0) side
        float eps_fd = 1e-5f;
        float th_fd  = theta_turn - sth_before * eps_fd;
        float c_fd   = cosf(th_fd);
        float s2_fd  = fmaxf(sinf(th_fd)*sinf(th_fd), 1e-8f);
        float Th_fd  = fmaxf(q2 - c_fd*c_fd*(b*b/s2_fd - a*a), 0.0f);
        float A_th   = Th_fd / eps_fd;
        float delta_theta = (-sth_before) * A_th * h_rem * h_rem * 0.25f;

        nx = { s_turn.r   + h_rem * k1r.r,
               theta_turn + delta_theta,
               s_turn.phi + h_rem * k1r.phi };
    }
    // Pole crossing: theta = 0 or pi is a coordinate singularity.  A geodesic
    // that reaches the polar axis passes THROUGH the pole onto the antipodal
    // meridian, so reflect theta back into range, jump phi by pi, and flip the
    // theta direction.  (The old hard clamp pinned near-axis rays against the
    // pole and dropped the phi+=pi jump, producing a doubled meridian along the
    // BH-axis direction in the lensed background.)
    if (nx.theta < 0.0f)                 { nx.theta = -nx.theta;                  nx.phi += CUDART_PI_F; sth = -sth; }
    else if (nx.theta > CUDART_PI_F)    { nx.theta = 2.0f*CUDART_PI_F - nx.theta; nx.phi += CUDART_PI_F; sth = -sth; }
    // Safety net for a pathological overshoot of more than pi in a single step.
    if (nx.theta < 1e-4f)                nx.theta = 1e-4f;
    if (nx.theta > CUDART_PI_F - 1e-4f) nx.theta = CUDART_PI_F - 1e-4f;
    return nx;
}

// ZAMO-tetrad initial conditions.
// Converts ZAMO-frame unit direction (n_r, n_th, n_ph) into conserved
// quantities b=L/E, q2=K/E^2 using the exact Kerr-Newman metric.
// The flat-space approx (b = r*sin(theta)*n_ph) is wrong near the BH by
// factor 1/alpha = 1/sqrt(Delta*Sigma/A), diverging at the horizon.
__device__ inline void kn_initial_conditions(
    float n_r, float n_th, float n_ph,
    float r0, float st, float ct,
    float M,  float a,  float Qbh,
    float& b, float& q2, float& sr, float& sth)
{
    // Use double precision to avoid catastrophic cancellation in (alpha - Omega*L)
    // near the ergosphere where the two terms nearly cancel in float32.
    double r0d = r0, std_ = st, ctd = ct, Md = M, ad = a, Qd = Qbh;
    double sig0d = r0d*r0d + ad*ad*ctd*ctd;
    double del0d = r0d*r0d - 2.0*Md*r0d + ad*ad + Qd*Qd;
    double rpa2d = r0d*r0d + ad*ad;
    double A0d   = fmax(rpa2d*rpa2d - del0d*ad*ad*std_*std_, 1e-10);
    double alpha0d = sqrt(fmax(del0d*sig0d/A0d, 0.0));
    double Omega0d = (2.0*Md*r0d - Qd*Qd)*ad / A0d;
    // ZAMO tetrad → conserved quantities.
    // Derivation: L = sqrt(g_φφ)*n_φ*E_ZAMO, E_ZAMO=(1-Ω*b)/α, b=L
    //   → b = S*n_φ / (α + Ω*S*n_φ)  where S = sinθ*sqrt(A/Σ)
    // Denominator is  α + Ω*L_aux  (NOT α − Ω*L_aux).
    // With + sign: R(r₀) = n_r²·E_ZAMO²·Δ·Σ ≥ 0 by construction → no sanity fix needed.
    double Laux  = fmax(fabs(std_), 1e-6) * sqrt(A0d/sig0d) * (double)n_ph;
    double Ecrdd = alpha0d + Omega0d * Laux;   // ← CORRECT sign: + not −
    if (Ecrdd < 1e-10) Ecrdd = 1e-10;
    double bd    = Laux / Ecrdd;
    b = (float)bd;

    double st2d = fmax(std_*std_, 1e-8);
    double Th0d = sig0d * (double)n_th * (double)n_th / (Ecrdd * Ecrdd);
    q2 = (float)fmax(Th0d + ctd*ctd*(bd*bd/st2d - ad*ad), 0.0);

    sr  = n_r  >= 0.0f ? 1.0f : -1.0f;
    sth = n_th >= 0.0f ? 1.0f : -1.0f;
    // No R(r₀) sanity fix: with the correct + sign R(r₀) = n_r²·E_ZAMO²·Δ·Σ ≥ 0 by construction.
}

// ── Camera basis → ray direction ──────────────────────────────────────────────
// Returns unit ray direction (wdx, wdy, wdz) for given NDC coords.
// Also outputs the camera spherical trig values needed for initial conditions.

__device__ void buildRayDir(float ndc_x, float ndc_y,
                             float aspect, float tan_hfov,
                             const RenderParams& p,
                             float& wdx, float& wdy, float& wdz)
{
    float rdx = ndc_x * aspect * tan_hfov;
    float rdy = ndc_y * tan_hfov;
    float rdz = 1.0f;

    float sin_ct = sinf(p.cam_theta), cos_ct = cosf(p.cam_theta);
    float sin_cp = sinf(p.cam_phi),   cos_cp = cosf(p.cam_phi);

    float fwx = -sin_ct*cos_cp, fwy = -sin_ct*sin_cp, fwz = -cos_ct;
    float upx = -cos_ct*cos_cp*cosf(p.cam_up_phi) + sinf(p.cam_up_phi)*(-sin_cp);
    float upy = -cos_ct*sin_cp*cosf(p.cam_up_phi) + sinf(p.cam_up_phi)*( cos_cp);
    float upz =  sin_ct*cosf(p.cam_up_phi);

    float rtx, rty, rtz;
    cross3(fwx,fwy,fwz, upx,upy,upz, rtx,rty,rtz);
    normalize3(rtx,rty,rtz);
    cross3(rtx,rty,rtz, fwx,fwy,fwz, upx,upy,upz);

    float cos_y  = cosf(p.cam_look_yaw),   sin_y  = sinf(p.cam_look_yaw);
    float cos_lp = cosf(p.cam_look_pitch),  sin_lp = sinf(p.cam_look_pitch);
    float fw2x = fwx*cos_y + rtx*sin_y, fw2y = fwy*cos_y + rty*sin_y, fw2z = fwz*cos_y + rtz*sin_y;
    float rt2x = rtx*cos_y - fwx*sin_y, rt2y = rty*cos_y - fwy*sin_y, rt2z = rtz*cos_y - fwz*sin_y;
    float fw3x = fw2x*cos_lp + upx*sin_lp, fw3y = fw2y*cos_lp + upy*sin_lp, fw3z = fw2z*cos_lp + upz*sin_lp;
    float up3x = upx*cos_lp - fw2x*sin_lp, up3y = upy*cos_lp - fw2y*sin_lp, up3z = upz*cos_lp - fw2z*sin_lp;

    wdx = rdx*rt2x + rdy*up3x + rdz*fw3x;
    wdy = rdx*rt2y + rdy*up3y + rdz*fw3y;
    wdz = rdx*rt2z + rdy*up3z + rdz*fw3z;
    normalize3(wdx, wdy, wdz);
}

// ── Single-ray trace ──────────────────────────────────────────────────────────
// Traces one ray given its NDC position and returns the composited color.
// n_steps: integration step budget (caller adjusts for critical rays).

__device__ float4 traceOnce(float ndc_x, float ndc_y,
                              float aspect, float tan_hfov,
                              cudaTextureObject_t bg_tex,
                              cudaTextureObject_t disk_tex,
                              const RenderParams p,
                              int n_steps)
{
    float M = p.bh.M, a = p.bh.a, Qbh = p.bh.Q;
    float r_h = KN::r_plus(M, a, Qbh);
    float r0  = p.cam_r;

    // Ray direction
    float wdx, wdy, wdz;
    buildRayDir(ndc_x, ndc_y, aspect, tan_hfov, p, wdx, wdy, wdz);

    // Initial conditions from ZAMO tetrad (exact Kerr-Newman)
    float st = sinf(p.cam_theta), ct = cosf(p.cam_theta);
    float sp = sinf(p.cam_phi),   cp = cosf(p.cam_phi);

    float n_r  =  st*cp * wdx + st*sp * wdy + ct * wdz;
    float n_th =  ct*cp * wdx + ct*sp * wdy - st * wdz;
    float n_ph = -sp * wdx + cp * wdy;

    float b, q2, sr, sth;
    kn_initial_conditions(n_r, n_th, n_ph, r0, st, ct, M, a, Qbh, b, q2, sr, sth);

    // Integrate
    KNState state = {r0, p.cam_theta, p.cam_phi};
    bool   absorbed     = false;
    bool   escaped      = false;
    bool   nan_detected = false;
    int    r_turns      = 0;
    int    theta_turns  = 0;
    float4 disk_col = {0,0,0,0};
    float  r_escape = fmaxf(500.0f * M, 2.0f * r0);

    for (int s = 0; s < n_steps; ++s) {
        float dist_h = state.r - r_h;
        if (dist_h <= 0.0f) { absorbed = true; break; }

        float cos_th = cosf(state.theta);
        float sig = state.r*state.r + a*a*cos_th*cos_th;
        // Allow steps up to 8× dist_h — prograde Kerr photon sphere can be very
        // close to the horizon (dist_h ≈ 0.12M for a=0.9M), which would otherwise
        // make each step tiny and exhaust the budget before covering the full orbit.
        // The absorbed-check below still terminates if we overshoot.
        float h = p.step_size * 0.5f
                  * fmaxf(fminf(state.r * 0.5f, dist_h * 8.0f), 1e-3f * M)
                  / fmaxf(sig, 1.0f);
        // Near the polar axis dphi/dlam ~ b/sin^2(theta) diverges; shrink the
        // step so the rapid azimuthal winding is resolved (otherwise a residual
        // meridian fold survives along the BH-axis direction).
        float sin_th = sqrtf(fmaxf(1.0f - cos_th*cos_th, 0.0f));
        if (sin_th < 0.2f) h *= fmaxf(sin_th / 0.2f, 0.05f);

        float old_r   = state.r;
        float old_th  = state.theta;
        float old_phi = state.phi;
        float old_sr  = sr;       // save before kn_rk4 may modify them
        float old_sth = sth;

        state = kn_rk4(state, b, q2, sr, sth, M, a, Qbh, h);

        if (!isfinite(state.r) || !isfinite(state.theta)) {
            nan_detected = true; absorbed = true; break;
        }
        if (sr  != old_sr)  r_turns++;
        if (sth != old_sth) theta_turns++;

        if (state.r < r_h * 1.001f) { absorbed = true; break; }
        if (state.r >= r_escape) { escaped = true; break; }

        if (p.show_disk && disk_col.w < 0.99f) {
            float z_old = old_r   * cosf(old_th);
            float z_new = state.r * cosf(state.theta);
            if (z_old * z_new < 0.0f) {
                // Bisect the crossing interval to reduce O(h) linear interpolation error.
                // 4 iterations narrow the bracket to h/16 of the original step.
                KNState bs    = {old_r, old_th, old_phi};
                float   bs_sr = old_sr, bs_sth = old_sth;
                float   bs_h  = h;
                float   bs_z  = z_old;
                for (int bi = 0; bi < 4; ++bi) {
                    bs_h *= 0.5f;
                    float t_sr = bs_sr, t_sth = bs_sth;
                    KNState mid = kn_rk4(bs, b, q2, t_sr, t_sth, M, a, Qbh, bs_h);
                    float z_m = mid.r * cosf(mid.theta);
                    if (bs_z * z_m >= 0.0f) {
                        // crossing in second half: advance lower bound
                        bs = mid; bs_sr = t_sr; bs_sth = t_sth; bs_z = z_m;
                    }
                    // else: crossing in first half, keep lower bound, halve step
                }
                // One final linear interpolation within the refined bracket
                float t_sr2 = bs_sr, t_sth2 = bs_sth;
                KNState bsend = kn_rk4(bs, b, q2, t_sr2, t_sth2, M, a, Qbh, bs_h);
                float z_e  = bsend.r * cosf(bsend.theta);
                float frac = fabsf(bs_z) / (fabsf(bs_z) + fabsf(z_e) + 1e-12f);
                float rc   = bs.r   + frac * (bsend.r   - bs.r);
                float phic = bs.phi + frac * (bsend.phi - bs.phi);

                float4 dc = diskColor(rc, phic, p.cam_phi, p.disk_inner, p.disk_outer, M, a);
                disk_col.x += dc.x * dc.w * (1.0f - disk_col.w);
                disk_col.y += dc.y * dc.w * (1.0f - disk_col.w);
                disk_col.z += dc.z * dc.w * (1.0f - disk_col.w);
                disk_col.w += dc.w         * (1.0f - disk_col.w);
            }
        }
    }

    bool polar_exit    = false;
    bool uv_invalid    = false;
    bool max_steps_hit = !absorbed && !escaped;

    float4 bg = {0, 0, 0, 1};
    float seam_dist = 0.5f; // distance from nearest u_tex=0/1 (with sky_lon_offset); 0.5 = far
    float u_raw = 0.0f, v_raw = 0.0f; // longitude/latitude without sky rotation (modes 4, 5)
    if (!absorbed && (escaped || max_steps_hit)) {
        sr = 1.0f;  // escaping ray is always moving outward

        float Rf, Tf;
        KN::computeGeodesicRHS(state.r, state.theta, M, a, Qbh, b, q2, Rf, Tf);
        float sig_f = state.r*state.r + a*a*cosf(state.theta)*cosf(state.theta);
        float isig  = 1.0f / fmaxf(sig_f, 1e-6f);
        float vr    = sr  * sqrtf(fmaxf(Rf, 0.0f)) * isig;
        float vth   = sth * sqrtf(fmaxf(Tf, 0.0f)) * isig;
        float vphi  = KN::dphidlam_sigma(state.r, state.theta, M, a, Qbh, b) * isig;

        float rf = state.r, thf = state.theta;
        // Normalize accumulated phi to [-π, π] to avoid sinf/cosf precision loss
        float phf = fmodf(state.phi, 2.0f * CUDART_PI_F);
        if (phf > CUDART_PI_F)  phf -= 2.0f * CUDART_PI_F;
        if (phf < -CUDART_PI_F) phf += 2.0f * CUDART_PI_F;
        float rho = sqrtf(rf*rf + a*a);
        float stf = sinf(thf), ctf = cosf(thf), spf = sinf(phf), cpf = cosf(phf);
        float ex = (rf/rho*stf*cpf)*vr + (rho*ctf*cpf)*vth + (-rho*stf*spf)*vphi;
        float ey = (rf/rho*stf*spf)*vr + (rho*ctf*spf)*vth + ( rho*stf*cpf)*vphi;
        float ez = ctf*vr - rf*stf*vth;
        float dir_mag2 = ex*ex + ey*ey + ez*ez;
        if (!isfinite(dir_mag2) || dir_mag2 < 1e-24f) {
            uv_invalid = true;
        } else {
            normalize3(ex, ey, ez);
            float r_xy_exit = sqrtf(ex*ex + ey*ey);
            polar_exit = (r_xy_exit <= 1e-6f);

            // Raw u/v — no sky rotation.  Longitude lives on S^1: u=0 ≡ u=1.
            // Used for procedural grid (mode 5) and cos/sin UV debug (mode 4).
            dirToUV(ex, ey, ez, u_raw, v_raw, phf);
            u_raw -= floorf(u_raw);

            // Rotated u/v — sky_lon_offset aligns the texture seam.
            // Used for texture background (mode 0) and seam visualizer (mode 3).
            float phi_shift = p.sky_lon_offset * 2.0f * CUDART_PI_F;
            float cs_s = cosf(phi_shift), sn_s = sinf(phi_shift);
            float ex_r = ex * cs_s - ey * sn_s;
            float ey_r = ex * sn_s + ey * cs_s;
            float u_tex, v_tex;
            dirToUV(ex_r, ey_r, ez, u_tex, v_tex, phf + phi_shift);
            u_tex -= floorf(u_tex);
            seam_dist = fminf(u_tex, 1.0f - u_tex);

            if (!isfinite(u_raw) || !isfinite(v_raw) ||
                !isfinite(u_tex) || !isfinite(v_tex)) {
                uv_invalid = true;
            } else if (p.debug_mode == 0) {
                bg = tex2D<float4>(bg_tex, u_tex, v_tex);
            } else if (p.debug_mode == 5) {
                // Shift u by half a cell so the atan2 seam (u=0=u=1) lands
                // at the midpoint of a grid cell (lon_d=0.5) instead of on a
                // grid line (lon_d=0).  Without this, the seam appears doubled
                // because both sides of the BH shadow have u≈0 and u≈1 of the
                // same physical meridian — both trigger the lon_d<half_w test.
                bg = proceduralGrid(u_raw + 0.5f / 24.0f, v_raw);
            }
        }
    }

    float alpha = disk_col.w;
    float4 result = {disk_col.x + bg.x*(1.0f-alpha),
                     disk_col.y + bg.y*(1.0f-alpha),
                     disk_col.z + bg.z*(1.0f-alpha),
                     1.0f};
    if (p.debug_mode == 3) {
        // Seam visualizer: magenta where u_tex≈0/1 (after sky_lon_offset), black elsewhere.
        if (seam_dist < 0.01f) result = {1.0f, 0.0f, 1.0f, 1.0f}; // magenta: near seam
        else                   result = {0.0f, 0.0f, 0.0f, 1.0f}; // black
    } else if (p.debug_mode == 4) {
        // UV false-color using S^1 encoding: R=cos(2πu), G=sin(2πu), B=v.
        // cos/sin are continuous and periodic — no discontinuity at atan2 seam.
        // Any remaining discontinuity points to a different UV anomaly.
        if (!escaped || absorbed)  result = {0.0f, 0.0f, 0.0f, 1.0f};
        else if (uv_invalid)       result = {1.0f, 1.0f, 0.0f, 1.0f};
        else {
            float cu = cosf(2.0f * CUDART_PI_F * u_raw) * 0.5f + 0.5f;
            float su = sinf(2.0f * CUDART_PI_F * u_raw) * 0.5f + 0.5f;
            result = {cu, su, v_raw, 1.0f};
        }
    } else if (p.debug_mode == 5) {
        // Procedural lat/lon grid.  bg was set to proceduralGrid(u_raw, v_raw) above,
        // so result already has disk composited over the grid.
        // Only override for UV errors on escaped rays.
        if (!absorbed && uv_invalid)
            result = {1.0f, 1.0f, 0.0f, 1.0f}; // yellow: UV error
        // else: result = disk + proceduralGrid (or disk + black for absorbed)
    } else if (p.debug_mode != 0) {
        if      (nan_detected)    result = {0.5f, 0.0f, 1.0f, 1.0f}; // purple: NaN/inf
        else if (max_steps_hit)   result = {1.0f, 0.0f, 0.0f, 1.0f}; // red: step budget
        else if (absorbed)        result = {0.0f, 0.0f, 0.0f, 1.0f}; // black: horizon
        else if (uv_invalid)      result = {1.0f, 1.0f, 0.0f, 1.0f}; // yellow: bad UV
        else if (polar_exit)      result = {0.0f, 1.0f, 0.0f, 1.0f}; // green: pole singularity
        else if (r_turns > 0)     result = {0.0f, 0.0f, 1.0f, 1.0f}; // blue: r turning point
        else if (theta_turns > 0) result = {0.0f, 1.0f, 1.0f, 1.0f}; // cyan: θ turning point
        // else: normal escaped → keep composited sky color
    }
    return result;
}

// ── Main kernel with smooth adaptive step budget ──────────────────────────────
//
// Step count follows a Lorentzian profile in |b_eff/b_crit − 1|:
//   n_steps = max_steps × (1 + 5 / (1 + (brat/0.06)²))
//   brat → 0   : ×6  (photon ring, max quality)
//   brat = 0.06: ×3.5
//   brat = 0.12: ×2
//   brat → ∞   : ×1  (far field, base quality)
//
// No per-frame spatial SSAA — the TAA Halton-jitter 32-frame blend provides
// continuous sub-pixel coverage without introducing zone-boundary seams.

__global__ void rayTracerKernel(float4* output,
                                 cudaTextureObject_t bg_tex,
                                 cudaTextureObject_t disk_tex,
                                 RenderParams p)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= p.width || py >= p.height) return;

    float aspect   = (float)p.width / (float)p.height;
    float tan_hfov = tanf(p.fov * 0.5f);

    // Center-pixel NDC with sub-pixel jitter for temporal anti-aliasing
    float ndc_x0 = ((px + 0.5f) / p.width  - 0.5f) * 2.0f + p.jitter_x;
    float ndc_y0 = ((py + 0.5f) / p.height - 0.5f) * 2.0f + p.jitter_y;

    // ── Quick b_eff estimate for center ray (no integration) ─────────────────
    float wdx, wdy, wdz;
    buildRayDir(ndc_x0, ndc_y0, aspect, tan_hfov, p, wdx, wdy, wdz);

    float st = sinf(p.cam_theta), ct = cosf(p.cam_theta);
    float sp = sinf(p.cam_phi),   cp = cosf(p.cam_phi);
    float r0 = p.cam_r, a = p.bh.a, M = p.bh.M;

    float Qbh = p.bh.Q;
    float n_r  =  st*cp * wdx + st*sp * wdy + ct * wdz;
    float n_th =  ct*cp * wdx + ct*sp * wdy - st * wdz;
    float n_ph = -sp * wdx + cp * wdy;
    float b, q2, sr_dummy, sth_dummy;
    kn_initial_conditions(n_r, n_th, n_ph, r0, st, ct, M, a, Qbh,
                          b, q2, sr_dummy, sth_dummy);

    float b_eff  = sqrtf(b*b + q2);

    // Kerr-accurate photon sphere critical impact parameters.
    // Schwarzschild uses b_crit = 3√3 M.  For Kerr the prograde and retrograde
    // photon spheres have different b_crit; using only the Schwarzschild value
    // under-budgets steps for prograde photons (b_eff ≈ 3.7 M at a/M=0.6),
    // causing the front/back hemisphere seam artifact.
    float b_crit_lo, b_crit_hi;
    if (fabsf(a) < 1e-5f * fmaxf(M, 1e-10f)) {
        b_crit_lo = b_crit_hi = 3.0f * sqrtf(3.0f) * M;
    } else {
        float a_n = fabsf(a) / fmaxf(M, 1e-10f);
        // Prograde photon sphere (smaller r) and retrograde (larger r)
        float r_psp = 2.0f*M*(1.0f + cosf(2.0f/3.0f * acosf(fmaxf(-1.0f, fminf(1.0f, -a_n)))));
        float r_psr = 2.0f*M*(1.0f + cosf(2.0f/3.0f * acosf(fmaxf(-1.0f, fminf(1.0f, +a_n)))));
        float a2 = a*a;
        b_crit_lo = fabsf((r_psp*r_psp*r_psp - 3.0f*M*r_psp*r_psp + a2*(r_psp+M))
                          / (a*(r_psp - M)));
        b_crit_hi = fabsf((r_psr*r_psr*r_psr - 3.0f*M*r_psr*r_psr + a2*(r_psr+M))
                          / (a*(r_psr - M)));
        if (b_crit_lo > b_crit_hi) { float t = b_crit_lo; b_crit_lo = b_crit_hi; b_crit_hi = t; }
    }
    // Photon is "near the photon sphere" if b_eff is close to EITHER b_crit
    float brat = fminf(fabsf(b_eff / fmaxf(b_crit_lo, 1e-4f) - 1.0f),
                       fabsf(b_eff / fmaxf(b_crit_hi, 1e-4f) - 1.0f));

    // ── Smooth adaptive step budget ───────────────────────────────────────────
    // Prograde Kerr photon sphere (a/M=0.6) needs ~4.6 orbits for brat≈0.
    // Each orbit ≈ 2100 integration steps → need ~9600 steps at the photon sphere.
    // Old multiplier of 6 (max=3600) only covered ~1.7 orbits → ~252° phi error.
    // Width widened to 0.10 so photons 10% outside b_crit also get enough budget.
    float inv_brat2 = 1.0f / (1.0f + (brat / 0.10f) * (brat / 0.10f));
    int   n_steps   = min((int)(p.max_steps * (1.0f + 15.0f * inv_brat2) + 0.5f),
                          p.max_steps * 16);

    float4 color = traceOnce(ndc_x0, ndc_y0, aspect, tan_hfov, bg_tex, disk_tex, p, n_steps);

    output[py * p.width + px] = color;
}

// ── Host launch ───────────────────────────────────────────────────────────────

void launchRayTracer(float4* d_output,
                     cudaTextureObject_t bg_tex,
                     cudaTextureObject_t disk_tex,
                     const RenderParams& params,
                     cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((params.width  + block.x - 1) / block.x,
              (params.height + block.y - 1) / block.y);
    rayTracerKernel<<<grid, block, 0, stream>>>(d_output, bg_tex, disk_tex, params);
}

// ── Temporal blend ────────────────────────────────────────────────────────────

__global__ void temporalBlendKernel(float4* accum, const float4* src, int n, float alpha)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 a = accum[i], s = src[i];
    float ia = 1.0f - alpha;
    accum[i] = { a.x*ia + s.x*alpha, a.y*ia + s.y*alpha, a.z*ia + s.z*alpha, 1.0f };
}

void launchTemporalBlend(float4* accum, const float4* newFrame,
                          int w, int h, float alpha, cudaStream_t stream)
{
    int n = w * h;
    temporalBlendKernel<<<(n + 255) / 256, 256, 0, stream>>>(accum, newFrame, n, alpha);
}
