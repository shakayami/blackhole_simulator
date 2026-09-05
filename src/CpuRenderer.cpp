#define _USE_MATH_DEFINES
#include "CpuRenderer.h"
#include "physics/KerrNewman.h"
#include <cmath>
#include <algorithm>
#include <thread>
#include <vector>

// All functions here mirror RayTracer.cu exactly, with __device__ removed and
// CUDA math intrinsics replaced by their std:: / C math equivalents.

static constexpr float CPU_PI = 3.14159265358979323846f;

// ── Vector helpers ────────────────────────────────────────────────────────────

static inline void cross3(float ax, float ay, float az,
                           float bx, float by, float bz,
                           float& rx, float& ry, float& rz) {
    rx = ay*bz - az*by; ry = az*bx - ax*bz; rz = ax*by - ay*bx;
}
static inline void normalize3(float& x, float& y, float& z) {
    float inv = 1.0f / std::sqrt(x*x + y*y + z*z);
    x *= inv; y *= inv; z *= inv;
}
static inline void dirToUV(float dx, float dy, float dz, float& u, float& v,
                             float fallback_phi = 0.0f) {
    float inv = 1.0f / std::sqrt(dx*dx + dy*dy + dz*dz);
    dx *= inv; dy *= inv; dz *= inv;
    float cz = std::max(-1.0f, std::min(1.0f, dz));
    v = std::acos(cz) / CPU_PI;
    float r_xy = std::sqrt(std::max(1.0f - cz*cz, 0.0f));
    if (r_xy > 1e-6f) {
        u = std::atan2(dy, dx) / (2.0f * CPU_PI) + 0.5f;
    } else {
        u = fallback_phi / (2.0f * CPU_PI) + 0.5f;
        u = u - std::floor(u);
    }
}


// Procedural lat/lon grid background (debug_mode 5).
// u in [0,1): longitude as circular variable on S^1.  u=0 ≡ u=1.
// v in [0,1]:  latitude (v=0 = north pole, v=1 = south pole).
//
// Longitude lines use circular distance min(frac, 1-frac) which is continuous
// and periodic on S^1 — the grid has no seam by construction.
static inline float4 proceduralGridCPU(float u, float v) {
    constexpr int   N_lon  = 24;
    constexpr int   N_lat  = 12;
    constexpr float half_w = 0.040f;

    float lf    = (u - std::floor(u)) * (float)N_lon;
    float lon_f = lf - std::floor(lf);
    float lon_d = std::min(lon_f, 1.0f - lon_f);

    float laf   = v * (float)N_lat;
    float lat_f = laf - std::floor(laf);
    float lat_d = std::min(lat_f, 1.0f - lat_f);

    if (lon_d < half_w || lat_d < half_w)
        return {0.80f, 0.85f, 1.00f, 1.0f};
    return {0.04f, 0.06f, 0.12f, 1.0f};
}

// ── Disk ─────────────────────────────────────────────────────────────────────

static inline float diskNoise(float r, float phi) {
    return 0.78f
         + 0.13f * std::sin(r * 2.3f + phi * 0.9f)
         + 0.09f * std::sin(r * 4.7f - phi * 2.1f);
}

static inline float4 diskColor(float r, float phi, float cam_phi,
                                float disk_inner, float disk_outer,
                                float M, float /*a*/) {
    if (r < disk_inner || r > disk_outer) return {0,0,0,0};
    float t    = (r - disk_inner) / std::max(disk_outer - disk_inner, 1e-6f);
    float temp = std::pow(std::max(1.0f - t, 0.005f), 0.75f);
    float R = 0.6f + 0.4f * temp;
    float G = temp * (2.0f - temp);
    float B = temp * temp * temp;
    float v   = std::min(std::sqrt(M / std::max(r, M * 1.05f)), 0.5f);
    float dop = std::max(0.4f, 1.0f + v * std::sin(cam_phi - phi));
    float boost = dop * dop;
    float nt  = 1.0f - std::sqrt(disk_inner / std::max(r, disk_inner * 1.001f));
    float lum = temp * boost * diskNoise(r, phi) * nt;
    float fade_i = 1.0f - std::exp(-(r - disk_inner) / std::max(disk_inner * 0.08f, 0.4f));
    float fade_o = 1.0f - std::exp(-(disk_outer - r) / std::max(disk_outer * 0.06f, 0.8f));
    float opacity = 0.80f * fade_i * fade_o;
    return {R*lum, G*lum, B*lum, opacity};
}

// ── Kerr-Newman geodesic (Mino time) ─────────────────────────────────────────

struct KNState { float r, theta, phi; };

static inline KNState kn_deriv(KNState s, float b, float q2,
                                float sr, float sth,
                                float M, float a, float Qbh) {
    float Rr, Tth;
    KN::computeGeodesicRHS(s.r, s.theta, M, a, Qbh, b, q2, Rr, Tth);
    return { sr  * std::sqrt(std::max(Rr,  0.0f)),
             sth * std::sqrt(std::max(Tth, 0.0f)),
             KN::dphidlam_sigma(s.r, s.theta, M, a, Qbh, b) };
}

static inline float thetaTurnBisect(float th_in, float th_out,
                                     float b, float q2, float a) {
    for (int i = 0; i < 12; ++i) {
        float th_mid = (th_in + th_out) * 0.5f;
        float c = std::cos(th_mid), s2 = std::max(1.0f - c*c, 1e-8f);
        if (q2 - c*c * (b*b/s2 - a*a) >= 0.0f) th_in  = th_mid;
        else                                      th_out = th_mid;
    }
    return (th_in + th_out) * 0.5f;
}

static inline float rTurnBisect(float r_in, float r_out,
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

static KNState kn_rk4(KNState s, float b, float q2,
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
        float r_turn = rTurnBisect(s.r, nx0.r, b, q2, M, a, Qbh);
        float dr_total = nx0.r - s.r;
        float frac_r = (std::abs(dr_total) > 1e-10f)
                       ? std::max(0.0f, std::min(1.0f, (r_turn - s.r) / dr_total))
                       : 0.5f;
        KNState s_rturn = { r_turn,
                            s.theta + frac_r * (nx0.theta - s.theta),
                            s.phi   + frac_r * (nx0.phi   - s.phi) };
        float sr_before = sr;
        sr = -sr;
        float h_rem_r = h * (1.0f - frac_r);
        KNState k1rt = kn_deriv(s_rturn, b, q2, sr, sth, M, a, Qbh);
        float eps_fd = 1e-5f;
        float r_fd   = r_turn - sr_before * eps_fd;
        float R_fd, dummy_t;
        KN::computeGeodesicRHS(r_fd, s_rturn.theta, M, a, Qbh, b, q2, R_fd, dummy_t);
        float A_r    = std::max(R_fd, 0.0f) / eps_fd;
        float delta_r = (-sr_before) * A_r * h_rem_r * h_rem_r * 0.25f;
        nx = { r_turn        + delta_r,
               s_rturn.theta + h_rem_r * k1rt.theta,
               s_rturn.phi   + h_rem_r * k1rt.phi };
    }

    if (Tn < 0.0f) {
        // Use nx0 for bisection bounds so a simultaneous r-turn does not corrupt
        // the theta-turn fraction computation.
        float theta_turn = thetaTurnBisect(s.theta, nx0.theta, b, q2, a);
        float dth_total  = nx0.theta - s.theta;
        float frac = (std::abs(dth_total) > 1e-10f)
                     ? std::max(0.0f, std::min(1.0f, (theta_turn - s.theta) / dth_total))
                     : 0.5f;
        KNState s_turn = { s.r   + frac * (nx0.r   - s.r),
                           theta_turn,
                           s.phi + frac * (nx0.phi - s.phi) };
        float sth_before = sth;
        sth = -sth;
        float h_rem = h * (1.0f - frac);
        KNState k1r = kn_deriv(s_turn, b, q2, sr, sth, M, a, Qbh);
        float eps_fd = 1e-5f;
        float th_fd  = theta_turn - sth_before * eps_fd;
        float c_fd   = std::cos(th_fd);
        float s2_fd  = std::max(std::sin(th_fd)*std::sin(th_fd), 1e-8f);
        float Th_fd  = std::max(q2 - c_fd*c_fd*(b*b/s2_fd - a*a), 0.0f);
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
    if (nx.theta < 0.0f)           { nx.theta = -nx.theta;              nx.phi += CPU_PI; sth = -sth; }
    else if (nx.theta > CPU_PI)    { nx.theta = 2.0f*CPU_PI - nx.theta; nx.phi += CPU_PI; sth = -sth; }
    // Safety net for a pathological overshoot of more than pi in a single step.
    if (nx.theta < 1e-4f)           nx.theta = 1e-4f;
    if (nx.theta > CPU_PI - 1e-4f)  nx.theta = CPU_PI - 1e-4f;
    return nx;
}

// ZAMO-tetrad initial conditions (identical derivation to GPU version).
static inline void kn_initial_conditions(
    float n_r, float n_th, float n_ph,
    float r0, float st, float ct,
    float M,  float a,  float Qbh,
    float& b, float& q2, float& sr, float& sth)
{
    double r0d = r0, std_ = st, ctd = ct, Md = M, ad = a, Qd = Qbh;
    double sig0d = r0d*r0d + ad*ad*ctd*ctd;
    double del0d = r0d*r0d - 2.0*Md*r0d + ad*ad + Qd*Qd;
    double rpa2d = r0d*r0d + ad*ad;
    double A0d   = std::max(rpa2d*rpa2d - del0d*ad*ad*std_*std_, 1e-10);
    double alpha0d = std::sqrt(std::max(del0d*sig0d/A0d, 0.0));
    double Omega0d = (2.0*Md*r0d - Qd*Qd)*ad / A0d;
    double Laux  = std::max(std::abs(std_), 1e-6) * std::sqrt(A0d/sig0d) * (double)n_ph;
    double Ecrdd = alpha0d + Omega0d * Laux;
    if (Ecrdd < 1e-10) Ecrdd = 1e-10;
    double bd    = Laux / Ecrdd;
    b = (float)bd;
    double st2d = std::max(std_*std_, 1e-8);
    double Th0d = sig0d * (double)n_th * (double)n_th / (Ecrdd * Ecrdd);
    q2 = (float)std::max(Th0d + ctd*ctd*(bd*bd/st2d - ad*ad), 0.0);
    sr  = n_r  >= 0.0f ? 1.0f : -1.0f;
    sth = n_th >= 0.0f ? 1.0f : -1.0f;
}

// ── Camera basis → ray direction ──────────────────────────────────────────────

static void buildRayDir(float ndc_x, float ndc_y,
                         float aspect, float tan_hfov,
                         const RenderParams& p,
                         float& wdx, float& wdy, float& wdz) {
    float rdx = ndc_x * aspect * tan_hfov;
    float rdy = ndc_y * tan_hfov;
    float rdz = 1.0f;

    float sin_ct = std::sin(p.cam_theta), cos_ct = std::cos(p.cam_theta);
    float sin_cp = std::sin(p.cam_phi),   cos_cp = std::cos(p.cam_phi);

    float fwx = -sin_ct*cos_cp, fwy = -sin_ct*sin_cp, fwz = -cos_ct;
    float upx = -cos_ct*cos_cp*std::cos(p.cam_up_phi) + std::sin(p.cam_up_phi)*(-sin_cp);
    float upy = -cos_ct*sin_cp*std::cos(p.cam_up_phi) + std::sin(p.cam_up_phi)*( cos_cp);
    float upz =  sin_ct*std::cos(p.cam_up_phi);

    float rtx, rty, rtz;
    cross3(fwx,fwy,fwz, upx,upy,upz, rtx,rty,rtz);
    normalize3(rtx,rty,rtz);
    cross3(rtx,rty,rtz, fwx,fwy,fwz, upx,upy,upz);

    float cos_y  = std::cos(p.cam_look_yaw),   sin_y  = std::sin(p.cam_look_yaw);
    float cos_lp = std::cos(p.cam_look_pitch),  sin_lp = std::sin(p.cam_look_pitch);
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

static float4 traceOnceCPU(float ndc_x, float ndc_y,
                             float aspect, float tan_hfov,
                             const CudaTexture& bg_tex,
                             const CudaTexture& disk_tex,
                             const RenderParams& p,
                             int n_steps) {
    float M = p.bh.M, a = p.bh.a, Qbh = p.bh.Q;
    float r_h = KN::r_plus(M, a, Qbh);
    float r0  = p.cam_r;

    float wdx, wdy, wdz;
    buildRayDir(ndc_x, ndc_y, aspect, tan_hfov, p, wdx, wdy, wdz);

    float st = std::sin(p.cam_theta), ct = std::cos(p.cam_theta);
    float sp = std::sin(p.cam_phi),   cp = std::cos(p.cam_phi);

    float n_r  =  st*cp * wdx + st*sp * wdy + ct * wdz;
    float n_th =  ct*cp * wdx + ct*sp * wdy - st * wdz;
    float n_ph = -sp * wdx + cp * wdy;

    float b, q2, sr, sth;
    kn_initial_conditions(n_r, n_th, n_ph, r0, st, ct, M, a, Qbh, b, q2, sr, sth);

    KNState state = {r0, p.cam_theta, p.cam_phi};
    bool   absorbed     = false;
    bool   escaped      = false;
    bool   nan_detected = false;
    int    r_turns      = 0;
    int    theta_turns  = 0;
    float4 disk_col = {0,0,0,0};
    float  r_escape = std::max(500.0f * M, 2.0f * r0);

    for (int s = 0; s < n_steps; ++s) {
        float dist_h = state.r - r_h;
        if (dist_h <= 0.0f) { absorbed = true; break; }

        float cos_th = std::cos(state.theta);
        float sig = state.r*state.r + a*a*cos_th*cos_th;
        float h = p.step_size * 0.5f
                  * std::max(std::min(state.r * 0.5f, dist_h * 8.0f), 1e-3f * M)
                  / std::max(sig, 1.0f);
        // Near the polar axis dphi/dlam ~ b/sin^2(theta) diverges; shrink the
        // step so the rapid azimuthal winding is resolved (otherwise a residual
        // meridian fold survives along the BH-axis direction).
        float sin_th = std::sqrt(std::max(1.0f - cos_th*cos_th, 0.0f));
        if (sin_th < 0.2f) h *= std::max(sin_th / 0.2f, 0.05f);

        float old_r   = state.r;
        float old_th  = state.theta;
        float old_phi = state.phi;
        float old_sr  = sr;
        float old_sth = sth;

        state = kn_rk4(state, b, q2, sr, sth, M, a, Qbh, h);

        if (!std::isfinite(state.r) || !std::isfinite(state.theta)) {
            nan_detected = true; absorbed = true; break;
        }
        if (sr  != old_sr)  r_turns++;
        if (sth != old_sth) theta_turns++;

        if (state.r < r_h * 1.001f) { absorbed = true; break; }
        if (state.r >= r_escape) { escaped = true; break; }

        if (p.show_disk && disk_col.w < 0.99f) {
            float z_old = old_r   * std::cos(old_th);
            float z_new = state.r * std::cos(state.theta);
            if (z_old * z_new < 0.0f) {
                KNState bs    = {old_r, old_th, old_phi};
                float   bs_sr = old_sr, bs_sth = old_sth;
                float   bs_h  = h;
                float   bs_z  = z_old;
                for (int bi = 0; bi < 4; ++bi) {
                    bs_h *= 0.5f;
                    float t_sr = bs_sr, t_sth = bs_sth;
                    KNState mid = kn_rk4(bs, b, q2, t_sr, t_sth, M, a, Qbh, bs_h);
                    float z_m = mid.r * std::cos(mid.theta);
                    if (bs_z * z_m >= 0.0f) {
                        bs = mid; bs_sr = t_sr; bs_sth = t_sth; bs_z = z_m;
                    }
                }
                float t_sr2 = bs_sr, t_sth2 = bs_sth;
                KNState bsend = kn_rk4(bs, b, q2, t_sr2, t_sth2, M, a, Qbh, bs_h);
                float z_e  = bsend.r * std::cos(bsend.theta);
                float frac = std::abs(bs_z) / (std::abs(bs_z) + std::abs(z_e) + 1e-12f);
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
    float seam_dist = 0.5f;
    float u_raw = 0.0f, v_raw = 0.0f;
    if (!absorbed && (escaped || max_steps_hit)) {
        sr = 1.0f;
        float Rf, Tf;
        KN::computeGeodesicRHS(state.r, state.theta, M, a, Qbh, b, q2, Rf, Tf);
        float sig_f = state.r*state.r + a*a*std::cos(state.theta)*std::cos(state.theta);
        float isig  = 1.0f / std::max(sig_f, 1e-6f);
        float vr    = sr  * std::sqrt(std::max(Rf, 0.0f)) * isig;
        float vth   = sth * std::sqrt(std::max(Tf, 0.0f)) * isig;
        float vphi  = KN::dphidlam_sigma(state.r, state.theta, M, a, Qbh, b) * isig;

        float rf = state.r, thf = state.theta;
        float phf = std::fmod(state.phi, 2.0f * CPU_PI);
        if (phf >  CPU_PI) phf -= 2.0f * CPU_PI;
        if (phf < -CPU_PI) phf += 2.0f * CPU_PI;
        float rho = std::sqrt(rf*rf + a*a);
        float stf = std::sin(thf), ctf = std::cos(thf), spf = std::sin(phf), cpf = std::cos(phf);
        float ex = (rf/rho*stf*cpf)*vr + (rho*ctf*cpf)*vth + (-rho*stf*spf)*vphi;
        float ey = (rf/rho*stf*spf)*vr + (rho*ctf*spf)*vth + ( rho*stf*cpf)*vphi;
        float ez = ctf*vr - rf*stf*vth;
        float dir_mag2 = ex*ex + ey*ey + ez*ez;
        if (!std::isfinite(dir_mag2) || dir_mag2 < 1e-24f) {
            uv_invalid = true;
        } else {
            normalize3(ex, ey, ez);
            float r_xy_exit = std::sqrt(ex*ex + ey*ey);
            polar_exit = (r_xy_exit <= 1e-6f);

            // Raw u/v — no sky rotation.  Longitude on S^1: u=0 ≡ u=1.
            dirToUV(ex, ey, ez, u_raw, v_raw, phf);
            u_raw = u_raw - std::floor(u_raw);

            // Rotated u/v — sky_lon_offset for texture alignment and seam visualizer.
            float phi_shift = p.sky_lon_offset * 2.0f * CPU_PI;
            float cs_s = std::cos(phi_shift), sn_s = std::sin(phi_shift);
            float ex_r = ex * cs_s - ey * sn_s;
            float ey_r = ex * sn_s + ey * cs_s;
            float u_tex, v_tex;
            dirToUV(ex_r, ey_r, ez, u_tex, v_tex, phf + phi_shift);
            u_tex = u_tex - std::floor(u_tex);
            seam_dist = std::min(u_tex, 1.0f - u_tex);

            if (!std::isfinite(u_raw) || !std::isfinite(v_raw) ||
                !std::isfinite(u_tex) || !std::isfinite(v_tex)) {
                uv_invalid = true;
            } else if (p.debug_mode == 0) {
                bg = bg_tex.sample(u_tex, v_tex);
            } else if (p.debug_mode == 5) {
                // Half-cell shift: seam at u=0=u=1 lands at lon_d=0.5 (midpoint),
                // not lon_d=0 (grid line).  Eliminates the doubled-line artifact.
                bg = proceduralGridCPU(u_raw + 0.5f / 24.0f, v_raw);
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
        // cos/sin are continuous across atan2 seam; any remaining discontinuity
        // indicates a different UV anomaly.
        if (!escaped || absorbed)  result = {0.0f, 0.0f, 0.0f, 1.0f};
        else if (uv_invalid)       result = {1.0f, 1.0f, 0.0f, 1.0f};
        else {
            float cu = std::cos(2.0f * CPU_PI * u_raw) * 0.5f + 0.5f;
            float su = std::sin(2.0f * CPU_PI * u_raw) * 0.5f + 0.5f;
            result = {cu, su, v_raw, 1.0f};
        }
    } else if (p.debug_mode == 5) {
        // Procedural lat/lon grid — bg was set to proceduralGridCPU above.
        if (!absorbed && uv_invalid)
            result = {1.0f, 1.0f, 0.0f, 1.0f};
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

// ── Adaptive step budget (same Lorentzian profile as GPU kernel) ──────────────

static int adaptiveSteps(float ndc_x, float ndc_y, float aspect, float tan_hfov,
                          const RenderParams& p) {
    float M = p.bh.M, a = p.bh.a;
    float wdx, wdy, wdz;
    buildRayDir(ndc_x, ndc_y, aspect, tan_hfov, p, wdx, wdy, wdz);
    float st = std::sin(p.cam_theta), ct = std::cos(p.cam_theta);
    float sp = std::sin(p.cam_phi),   cp = std::cos(p.cam_phi);
    float n_r  =  st*cp*wdx + st*sp*wdy + ct*wdz;
    float n_th =  ct*cp*wdx + ct*sp*wdy - st*wdz;
    float n_ph = -sp*wdx + cp*wdy;
    float b, q2, sr_d, sth_d;
    kn_initial_conditions(n_r, n_th, n_ph, p.cam_r, st, ct, M, a, p.bh.Q, b, q2, sr_d, sth_d);
    float b_eff = std::sqrt(b*b + q2);
    float b_crit_lo, b_crit_hi;
    if (std::abs(a) < 1e-5f * std::max(M, 1e-10f)) {
        b_crit_lo = b_crit_hi = 3.0f * std::sqrt(3.0f) * M;
    } else {
        float a_n = std::abs(a) / std::max(M, 1e-10f);
        float r_psp = 2.0f*M*(1.0f + std::cos(2.0f/3.0f * std::acos(std::max(-1.0f, std::min(1.0f, -a_n)))));
        float r_psr = 2.0f*M*(1.0f + std::cos(2.0f/3.0f * std::acos(std::max(-1.0f, std::min(1.0f, +a_n)))));
        float a2 = a*a;
        b_crit_lo = std::abs((r_psp*r_psp*r_psp - 3.0f*M*r_psp*r_psp + a2*(r_psp+M)) / (a*(r_psp-M)));
        b_crit_hi = std::abs((r_psr*r_psr*r_psr - 3.0f*M*r_psr*r_psr + a2*(r_psr+M)) / (a*(r_psr-M)));
        if (b_crit_lo > b_crit_hi) std::swap(b_crit_lo, b_crit_hi);
    }
    float brat = std::min(std::abs(b_eff / std::max(b_crit_lo, 1e-4f) - 1.0f),
                          std::abs(b_eff / std::max(b_crit_hi, 1e-4f) - 1.0f));
    float inv_brat2 = 1.0f / (1.0f + (brat / 0.10f) * (brat / 0.10f));
    return std::min((int)(p.max_steps * (1.0f + 15.0f * inv_brat2) + 0.5f),
                    p.max_steps * 16);
}

// ── Public launch functions ───────────────────────────────────────────────────

void launchRayTracerCPU(float4*             output,
                         const CudaTexture& bg_tex,
                         const CudaTexture& disk_tex,
                         const RenderParams& params) {
    int W = params.width, H = params.height;
    float aspect   = (float)W / (float)H;
    float tan_hfov = std::tan(params.fov * 0.5f);

    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads < 1) nthreads = 1;

    std::vector<std::thread> threads(nthreads);
    for (unsigned t = 0; t < nthreads; ++t) {
        threads[t] = std::thread([=, &bg_tex, &disk_tex, &params]() {
            for (int py = (int)t; py < H; py += (int)nthreads) {
                for (int px = 0; px < W; ++px) {
                    float ndc_x = ((px + 0.5f) / W - 0.5f) * 2.0f + params.jitter_x;
                    float ndc_y = ((py + 0.5f) / H - 0.5f) * 2.0f + params.jitter_y;
                    int n_steps = adaptiveSteps(ndc_x, ndc_y, aspect, tan_hfov, params);
                    output[py * W + px] = traceOnceCPU(ndc_x, ndc_y, aspect, tan_hfov,
                                                        bg_tex, disk_tex, params, n_steps);
                }
            }
        });
    }
    for (auto& th : threads) th.join();
}

void launchTemporalBlendCPU(float4* accum, const float4* src,
                             int w, int h, float alpha) {
    int n = w * h;
    float ia = 1.0f - alpha;
    for (int i = 0; i < n; ++i) {
        float4 a4 = accum[i], s = src[i];
        accum[i] = { a4.x*ia + s.x*alpha, a4.y*ia + s.y*alpha, a4.z*ia + s.z*alpha, 1.0f };
    }
}
