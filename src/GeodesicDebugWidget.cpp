#include "GeodesicDebugWidget.h"
#include "physics/KerrNewman.h"
#include <QPainter>
#include <QPen>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QLabel>
#include <QSplitter>
#include <QFont>
#include <QDebug>
#include <QTabWidget>
#include <cmath>
#include <algorithm>

// ── CPU geodesic helpers (mirror GPU code exactly, including bisection) ────────

struct CPUState { float r, theta, phi; };

static CPUState kn_deriv_cpu(CPUState s, float b, float q2,
                               float sr, float sth, float M, float a, float Q) {
    float Rr, Tth;
    KN::computeGeodesicRHS(s.r, s.theta, M, a, Q, b, q2, Rr, Tth);
    return { sr  * std::sqrt(std::max(Rr,  0.0f)),
             sth * std::sqrt(std::max(Tth, 0.0f)),
             KN::dphidlam_sigma(s.r, s.theta, M, a, Q, b) };
}

static float thetaTurn_cpu(float th_in, float th_out, float b, float q2, float a) {
    for (int i = 0; i < 12; ++i) {
        float th_mid = (th_in + th_out) * 0.5f;
        float c = std::cos(th_mid), s2 = std::max(1.0f - c*c, 1e-8f);
        if (q2 - c*c*(b*b/s2 - a*a) >= 0.0f) th_in  = th_mid;
        else                                    th_out = th_mid;
    }
    return (th_in + th_out) * 0.5f;
}

static float rTurnBisect_cpu(float r_in, float r_out,
                               float b, float q2,
                               float M, float a, float Q) {
    for (int i = 0; i < 12; ++i) {
        float r_mid = (r_in + r_out) * 0.5f;
        float Rm, dummy;
        KN::computeGeodesicRHS(r_mid, 1.0f, M, a, Q, b, q2, Rm, dummy);
        if (Rm >= 0.0f) r_in  = r_mid;
        else            r_out = r_mid;
    }
    return (r_in + r_out) * 0.5f;
}

// Compute g^{μν} p_μ p_ν for null check (should be 0).
// p_t=-1, p_φ=b are the conserved covariant momenta (E=1 normalization).
// p_r = sr*sqrt(R)/Δ,  p_θ = sth*sqrt(Θ)  from Mino-time kinematics.
static float nullResidual(float r, float theta, float b, float q2,
                           float M, float a, float Q) {
    float Rr, Tth;
    KN::computeGeodesicRHS(r, theta, M, a, Q, b, q2, Rr, Tth);
    float sig = r*r + a*a*std::cos(theta)*std::cos(theta);
    float del = std::max(r*r - 2.f*M*r + a*a + Q*Q, 1e-10f);
    float sin2 = std::max(std::sin(theta)*std::sin(theta), 1e-8f);
    float A    = (r*r+a*a)*(r*r+a*a) - del*a*a*sin2;

    float pr  = std::sqrt(std::max(Rr,  0.f)) / del;  // p_r (covariant)
    float pth = std::sqrt(std::max(Tth, 0.f));         // p_θ (covariant)

    // inverse metric KN
    float gtt   = -A  / (sig*del);
    float gtphi =  (2.f*M*r - Q*Q)*a / (sig*del);
    float gphph =  (del - a*a*sin2) / (sig*del*sin2);
    float grr   =  del / sig;
    float gthth =  1.f / sig;

    float res = grr*pr*pr + gthth*pth*pth
              + gtt*1.f        // p_t = -1, so p_t^2 = 1
              + 2.f*gtphi*(-1.f)*b
              + gphph*b*b;
    return std::abs(res);
}

// Full CPU RK4 integration with theta bisection (identical logic to GPU)
// max_phi_wind: stop after accumulating this many radians of |Δφ| (0 = unlimited)
static GeodesicRay integrateCPU(float b, float q2, float sr, float sth,
                                  float r0, float theta0, float phi0,
                                  float M, float a, float Q,
                                  int max_steps, float step_size,
                                  float max_phi_wind, float sky_lon_offset,
                                  QColor color, float ndc_x)
{
    GeodesicRay ray;
    ray.b = b; ray.q2 = q2; ray.color = color; ray.ndc_x = ndc_x;
    ray.absorbed = false; ray.winding_terminated = false; ray.escaped = false;
    ray.phi_total = 0.f; ray.n_theta_turns = 0; ray.n_r_turns = 0;
    ray.Theta_min = 1e10f; ray.r_min_turning = 1e10f; ray.null_max = 0.f;
    ray.exit_u = ray.exit_v = 0.5f;
    ray.exit_ex = ray.exit_ey = ray.exit_ez = 0.f;
    ray.exit_rho_xy = 0.f; ray.exit_atan2 = 0.f;

    float r = r0, theta = theta0, phi = phi0;
    float phi_prev = phi0;
    float r_h      = KN::r_plus(M, a, Q);
    float r_escape = std::max(500.f * M, 2.f * r0);

    for (int s = 0; s < max_steps; ++s) {
        // ── Record current state ──────────────────────────────────────────────
        float Rr, Tth;
        KN::computeGeodesicRHS(r, theta, M, a, Q, b, q2, Rr, Tth);
        float nr_ = nullResidual(r, theta, b, q2, M, a, Q);

        GeodesicPt pt;
        pt.r = r; pt.theta = theta; pt.phi = phi;
        pt.Theta_val = Tth; pt.R_val = Rr;
        pt.null_resid = nr_;
        pt.sth_sign = (sth > 0) ? 1 : -1;
        pt.sr_sign  = (sr  > 0) ? 1 : -1;
        pt.turning   = false;
        pt.r_turning = false;
        ray.pts.push_back(pt);

        ray.Theta_min = std::min(ray.Theta_min, Tth);
        ray.null_max  = std::max(ray.null_max,  nr_);

        // ── Termination ───────────────────────────────────────────────────────
        if (r <= r_h) { ray.absorbed = true; break; }
        if (r >= r_escape) { ray.escaped = true; break; }
        ray.phi_total += std::abs(phi - phi_prev);
        phi_prev = phi;
        if (max_phi_wind > 0.f && ray.phi_total >= max_phi_wind) {
            ray.winding_terminated = true;
            break;
        }

        // ── Adaptive step (same formula as GPU) ───────────────────────────────
        float cos_th = std::cos(theta);
        float sig    = r*r + a*a*cos_th*cos_th;
        float dist_h = r - r_h;
        float h = step_size * 0.5f
                  * std::max(std::min(r*0.5f, dist_h*8.f), 1e-3f*M)
                  / std::max(sig, 1.f);
        // Near the polar axis dphi/dlam ~ b/sin^2(theta) diverges; shrink the
        // step so the rapid azimuthal winding is resolved (otherwise a residual
        // meridian fold survives along the BH-axis direction).
        float sin_th = std::sqrt(std::max(1.0f - cos_th*cos_th, 0.f));
        if (sin_th < 0.2f) h *= std::max(sin_th / 0.2f, 0.05f);

        // ── RK4 ──────────────────────────────────────────────────────────────
        CPUState st0 = {r, theta, phi};
        CPUState k1  = kn_deriv_cpu(st0,                                               b, q2, sr, sth, M, a, Q);
        CPUState s2  = {r+.5f*h*k1.r, theta+.5f*h*k1.theta, phi+.5f*h*k1.phi};
        CPUState k2  = kn_deriv_cpu(s2,                                                b, q2, sr, sth, M, a, Q);
        CPUState s3  = {r+.5f*h*k2.r, theta+.5f*h*k2.theta, phi+.5f*h*k2.phi};
        CPUState k3  = kn_deriv_cpu(s3,                                                b, q2, sr, sth, M, a, Q);
        CPUState s4  = {r+h*k3.r, theta+h*k3.theta, phi+h*k3.phi};
        CPUState k4  = kn_deriv_cpu(s4,                                                b, q2, sr, sth, M, a, Q);

        float nr  = r     + h/6.f*(k1.r     + 2*k2.r     + 2*k3.r     + k4.r);
        float nth = theta + h/6.f*(k1.theta + 2*k2.theta + 2*k3.theta + k4.theta);
        float nph = phi   + h/6.f*(k1.phi   + 2*k2.phi   + 2*k3.phi   + k4.phi);

        float Rn, Tn;
        KN::computeGeodesicRHS(nr, nth, M, a, Q, b, q2, Rn, Tn);

        // Save original RK4 result before turning-point correction
        float nr0 = nr, nth0 = nth, nph0 = nph;

        if (Rn < 0.f) {
            float r_turn = rTurnBisect_cpu(r, nr0, b, q2, M, a, Q);
            float dr_total = nr0 - r;
            float frac_r = std::abs(dr_total) > 1e-10f
                           ? std::max(0.f, std::min(1.f, (r_turn - r) / dr_total))
                           : 0.5f;
            float rth_theta = theta + frac_r * (nth0 - theta);
            float rth_phi   = phi   + frac_r * (nph0 - phi);
            CPUState s_rturn = {r_turn, rth_theta, rth_phi};

            float sr_before = sr;
            sr = -sr;
            float h_rem_r = h * (1.f - frac_r);
            CPUState k1rt = kn_deriv_cpu(s_rturn, b, q2, sr, sth, M, a, Q);

            constexpr float EPS_FD_R = 1e-5f;
            float r_fd = r_turn - sr_before * EPS_FD_R;
            float R_fd, dummy_t;
            KN::computeGeodesicRHS(r_fd, rth_theta, M, a, Q, b, q2, R_fd, dummy_t);
            float A_r    = std::max(R_fd, 0.f) / EPS_FD_R;
            float delta_r = (-sr_before) * A_r * h_rem_r * h_rem_r * 0.25f;

            nr  = r_turn   + delta_r;
            nth = rth_theta + h_rem_r * k1rt.theta;
            nph = rth_phi   + h_rem_r * k1rt.phi;

            ray.pts.back().r_turning = true;
            ray.n_r_turns++;
            ray.r_min_turning = std::min(ray.r_min_turning, r_turn);
        }

        if (Tn < 0.f) {
            // Use original RK4 theta values for bisection bounds so a simultaneous
            // r-turn does not corrupt the theta-turn fraction computation.
            float theta_turn = thetaTurn_cpu(theta, nth0, b, q2, a);
            float dth_total  = nth0 - theta;
            float frac = std::abs(dth_total) > 1e-10f
                         ? std::max(0.f, std::min(1.f, (theta_turn - theta) / dth_total))
                         : 0.5f;

            float r_turn_th = r   + frac*(nr0  - r);
            float phi_turn  = phi + frac*(nph0 - phi);

            ray.pts.back().turning = true;
            ray.n_theta_turns++;
            float sth_before = sth;
            sth = -sth;

            float h_rem = h * (1.f - frac);
            CPUState st_turn = {r_turn_th, theta_turn, phi_turn};

            // r, phi: standard Euler (derivatives non-zero at theta_turn)
            CPUState k1r = kn_deriv_cpu(st_turn, b, q2, sr, sth, M, a, Q);
            nr  = st_turn.r   + h_rem * k1r.r;
            nph = st_turn.phi + h_rem * k1r.phi;

            // theta: analytic quadratic advance from turning point
            constexpr float EPS_FD = 1e-5f;
            float th_fd  = theta_turn - sth_before * EPS_FD;
            float c_fd   = std::cos(th_fd);
            float s2_fd  = std::max(std::sin(th_fd)*std::sin(th_fd), 1e-8f);
            float Th_fd  = std::max(q2 - c_fd*c_fd*(b*b/s2_fd - a*a), 0.f);
            float A_th   = Th_fd / EPS_FD;
            nth = theta_turn + (-sth_before) * A_th * h_rem * h_rem * 0.25f;
        }

        constexpr float PI = 3.14159265358979f;
        // Pole crossing: theta = 0 or pi is a coordinate singularity.  The photon
        // passes THROUGH the pole onto the antipodal meridian, so reflect theta,
        // jump phi by pi, and flip the theta direction.  (The old hard clamp
        // pinned near-axis rays and dropped the phi+=pi jump, producing a doubled
        // meridian along the BH-axis direction.)
        if (nth < 0.f)        { nth = -nth;        nph += PI; sth = -sth; }
        else if (nth > PI)    { nth = 2.f*PI - nth; nph += PI; sth = -sth; }
        if (nth < 1e-4f)      nth = 1e-4f;
        if (nth > PI - 1e-4f) nth = PI - 1e-4f;

        r = nr; theta = nth; phi = nph;
    }

    // ── Compute exit sky UV (only for rays that actually escaped) ────────────
    if (!ray.absorbed && ray.escaped && !ray.pts.empty()) {
        sr = 1.f;
        const GeodesicPt& last = ray.pts.back();
        float Rf, Tf;
        KN::computeGeodesicRHS(last.r, last.theta, M, a, Q, b, q2, Rf, Tf);
        float sig_f = last.r*last.r + a*a*std::cos(last.theta)*std::cos(last.theta);
        float isig  = 1.f / std::max(sig_f, 1e-6f);
        float vr    =  sr  * std::sqrt(std::max(Rf, 0.f)) * isig;
        float vth   =  sth * std::sqrt(std::max(Tf, 0.f)) * isig;
        float vphi  =  KN::dphidlam_sigma(last.r, last.theta, M, a, Q, b) * isig;

        constexpr float PI = 3.14159265358979f;
        float phf = std::fmod(last.phi, 2.f*PI);
        if (phf >  PI) phf -= 2.f*PI;
        if (phf < -PI) phf += 2.f*PI;
        float rho = std::sqrt(last.r*last.r + a*a);
        float stf = std::sin(last.theta), ctf = std::cos(last.theta);
        float spf = std::sin(phf),        cpf = std::cos(phf);

        float ex = (last.r/rho*stf*cpf)*vr + (rho*ctf*cpf)*vth + (-rho*stf*spf)*vphi;
        float ey = (last.r/rho*stf*spf)*vr + (rho*ctf*spf)*vth + ( rho*stf*cpf)*vphi;
        float ez =  ctf*vr - last.r*stf*vth;
        float inv = 1.f / std::max(std::sqrt(ex*ex+ey*ey+ez*ez), 1e-10f);
        ex *= inv; ey *= inv; ez *= inv;

        // Store normalized exit direction for diagnostics
        ray.exit_ex = ex; ray.exit_ey = ey; ray.exit_ez = ez;
        ray.exit_rho_xy = std::sqrt(ex*ex + ey*ey);
        ray.exit_atan2  = std::atan2(ey, ex);

        float cz = std::max(-1.f, std::min(1.f, ez));
        ray.exit_v = std::acos(cz) / PI;

        float r_xy = ray.exit_rho_xy;
        // Near poles use escape φ as u fallback (continuity with neighbor pixels);
        // far from poles use atan2 of the exit direction.
        float raw_u;
        if (r_xy > 1e-6f) {
            raw_u = ray.exit_atan2 / (2.f*PI) + 0.5f;
        } else {
            raw_u = phf / (2.f*PI) + 0.5f;
            raw_u = raw_u - std::floor(raw_u);
        }
        // Apply sky longitude offset to rotate seam away from BH shadow boundary
        raw_u += sky_lon_offset;
        ray.exit_u = raw_u - std::floor(raw_u);
    }
    return ray;
}

// ── buildRayDir + kn_initial_conditions (CPU copies of GPU functions) ─────────

static void buildRayDir_cpu(float ndc_x, float ndc_y,
                              float aspect, float tan_hfov,
                              const RenderParams& p,
                              float& wdx, float& wdy, float& wdz)
{
    float rdx = ndc_x * aspect * tan_hfov, rdy = ndc_y * tan_hfov, rdz = 1.f;

    float sin_ct = std::sin(p.cam_theta), cos_ct = std::cos(p.cam_theta);
    float sin_cp = std::sin(p.cam_phi),   cos_cp = std::cos(p.cam_phi);

    float fwx = -sin_ct*cos_cp, fwy = -sin_ct*sin_cp, fwz = -cos_ct;
    float upx = -cos_ct*cos_cp*std::cos(p.cam_up_phi) + std::sin(p.cam_up_phi)*(-sin_cp);
    float upy = -cos_ct*sin_cp*std::cos(p.cam_up_phi) + std::sin(p.cam_up_phi)*( cos_cp);
    float upz =  sin_ct*std::cos(p.cam_up_phi);

    float rtx = fwy*upz - fwz*upy, rty = fwz*upx - fwx*upz, rtz = fwx*upy - fwy*upx;
    float inv = 1.f / std::sqrt(rtx*rtx + rty*rty + rtz*rtz);
    rtx *= inv; rty *= inv; rtz *= inv;
    upx = rty*fwz - rtz*fwy; upy = rtz*fwx - rtx*fwz; upz = rtx*fwy - rty*fwx;

    float cos_y  = std::cos(p.cam_look_yaw),   sin_y  = std::sin(p.cam_look_yaw);
    float cos_lp = std::cos(p.cam_look_pitch),  sin_lp = std::sin(p.cam_look_pitch);
    float fw2x = fwx*cos_y + rtx*sin_y, fw2y = fwy*cos_y + rty*sin_y, fw2z = fwz*cos_y + rtz*sin_y;
    float rt2x = rtx*cos_y - fwx*sin_y, rt2y = rty*cos_y - fwy*sin_y, rt2z = rtz*cos_y - fwz*sin_y;
    float fw3x = fw2x*cos_lp + upx*sin_lp, fw3y = fw2y*cos_lp + upy*sin_lp, fw3z = fw2z*cos_lp + upz*sin_lp;
    float up3x = upx*cos_lp - fw2x*sin_lp, up3y = upy*cos_lp - fw2y*sin_lp, up3z = upz*cos_lp - fw2z*sin_lp;

    wdx = rdx*rt2x + rdy*up3x + rdz*fw3x;
    wdy = rdx*rt2y + rdy*up3y + rdz*fw3y;
    wdz = rdx*rt2z + rdy*up3z + rdz*fw3z;
    inv = 1.f / std::sqrt(wdx*wdx + wdy*wdy + wdz*wdz);
    wdx *= inv; wdy *= inv; wdz *= inv;
}

static void kn_initial_conditions_cpu(
    float n_r, float n_th, float n_ph,
    float r0, float st, float ct,
    float M, float a, float Qbh,
    float& b, float& q2, float& sr, float& sth)
{
    // Use double precision to avoid catastrophic cancellation in (alpha - Omega*L)
    // near the ergosphere where the two terms nearly cancel in float32.
    double r0d = r0, std_ = st, ctd = ct, Md = M, ad = a, Qd = Qbh;
    double sig0d = r0d*r0d + ad*ad*ctd*ctd;
    double del0d = r0d*r0d - 2.0*Md*r0d + ad*ad + Qd*Qd;
    double rpa2d = r0d*r0d + ad*ad;
    double A0d   = std::max(rpa2d*rpa2d - del0d*ad*ad*std_*std_, 1e-10);
    double alpha0d = std::sqrt(std::max(del0d*sig0d/A0d, 0.0));
    double Omega0d = (2.0*Md*r0d - Qd*Qd)*ad / A0d;
    // ZAMO tetrad → conserved quantities.
    // Derivation: L = sqrt(g_φφ)*n_φ*E_ZAMO, E_ZAMO=(1-Ω*b)/α, b=L
    //   → b = S*n_φ / (α + Ω*S*n_φ)  where S = sinθ*sqrt(A/Σ)
    // Denominator is  α + Ω*L_aux  (NOT α − Ω*L_aux).
    // With + sign: R(r₀) = n_r²·E_ZAMO²·Δ·Σ ≥ 0 by construction → no sanity fix needed.
    double Laux  = std::max(std::abs(std_), 1e-6) * std::sqrt(A0d/sig0d) * (double)n_ph;
    double Ecrdd = alpha0d + Omega0d * Laux;   // ← CORRECT sign: + not −
    if (Ecrdd < 1e-10) Ecrdd = 1e-10;
    double bd    = Laux / Ecrdd;
    b  = (float)bd;
    double st2d  = std::max(std_*std_, 1e-8);
    double Th0d  = sig0d * (double)n_th * (double)n_th / (Ecrdd * Ecrdd);
    q2 = (float)std::max(Th0d + ctd*ctd*(bd*bd/st2d - ad*ad), 0.0);
    sr  = n_r  >= 0.f ? 1.f : -1.f;
    sth = n_th >= 0.f ? 1.f : -1.f;
}

// ── GeodesicView ─────────────────────────────────────────────────────────────

GeodesicView::GeodesicView(QWidget* parent) : QWidget(parent) {
    setMinimumSize(400, 300);
}

void GeodesicView::setData(const std::vector<GeodesicRay>& rays,
                            float r_h, float r_ph_lo, float r_ph_hi,
                            float r_isco, float M,
                            float cam_r, float cam_theta, float cam_phi,
                            bool isVertical)
{
    m_rays = rays;
    m_r_h = r_h; m_r_ph_lo = r_ph_lo; m_r_ph_hi = r_ph_hi;
    m_r_isco = r_isco; m_M = M;
    m_cam_r = cam_r; m_cam_theta = cam_theta; m_cam_phi = cam_phi;
    m_isVertical = isVertical;
    update();
}

void GeodesicView::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    int third = width() / 3;
    drawOverhead( p, QRect(0,         0, third,              height()));
    drawSideView( p, QRect(third,     0, third,              height()));
    drawThetaPlot(p, QRect(2 * third, 0, width() - 2*third, height()));
}

void GeodesicView::drawOverhead(QPainter& p, const QRect& rc) {
    p.fillRect(rc, Qt::black);

    // Base scale: fit the furthest point across all traced paths
    float maxR = 15.f * m_M;
    for (const auto& ray : m_rays)
        for (const auto& pt  : ray.pts)
            maxR = std::max(maxR, std::min(pt.r, 300.f * m_M));
    maxR = std::min(maxR, 200.f * m_M);

    float cx = rc.left() + rc.width()  * 0.5f;
    float cy = rc.top()  + rc.height() * 0.5f;
    float base_scale   = std::min(rc.width(), rc.height()) * 0.43f / maxR;
    float total_scale  = base_scale * m_zoom;

    // World → screen.
    // screen_x = cx + world_x * total_scale + pan_x
    // screen_y = cy - world_y * total_scale + pan_y   (y flipped: world up = screen up)
    auto toScreen = [&](float r, float phi) -> QPointF {
        return { cx + r * std::cos(phi) * total_scale + m_pan.x(),
                 cy - r * std::sin(phi) * total_scale + m_pan.y() };
    };

    // BH origin in screen coords (affected by pan)
    QPointF bhCtr(cx + m_pan.x(), cy + m_pan.y());

    auto drawRing = [&](float r, QColor c, float w = 1.f, bool fill = false) {
        float px = r * total_scale;
        p.setPen(QPen(c, w));
        p.setBrush(fill ? QBrush(c) : Qt::NoBrush);
        p.drawEllipse(bhCtr, px, px);
    };

    // Ergosphere (equatorial: r_ergo ≈ 2M for Q=0)
    float r_ergo = m_M + std::sqrt(std::max(m_M*m_M - 0.f, 0.f));
    if (r_ergo > m_r_h * 1.01f)
        drawRing(r_ergo,  QColor(120, 0, 0),   1.5f);
    drawRing(m_r_isco,    QColor(0, 200, 80),   1.0f);
    if (std::abs(m_r_ph_hi - m_r_ph_lo) > 0.05f * m_M) {
        drawRing(m_r_ph_lo, QColor(240, 200, 0), 1.5f);
        drawRing(m_r_ph_hi, QColor(240, 200, 0), 1.5f);
    } else {
        drawRing(m_r_ph_lo, QColor(240, 200, 0), 1.5f);
    }
    drawRing(m_r_h, QColor(40, 40, 40), 2.0f, true);

    // Camera dot (equatorial projection: r_eq = r*sin(θ))
    QPointF cam_pt = toScreen(m_cam_r * std::sin(m_cam_theta), m_cam_phi);
    p.setPen(QPen(Qt::white, 2));
    p.setBrush(Qt::white);
    p.drawEllipse(cam_pt, 4, 4);

    // Geodesic paths: phi accumulates → spiral orbit visible
    p.setClipRect(rc);
    for (const auto& ray : m_rays) {
        if (ray.pts.empty()) continue;
        p.setPen(QPen(ray.color, 1.2f));
        p.setBrush(Qt::NoBrush);
        QPolygonF poly;
        for (const auto& pt : ray.pts)
            poly << toScreen(pt.r * std::sin(pt.theta), pt.phi);
        p.drawPolyline(poly);

        for (const auto& pt : ray.pts) {
            if (pt.turning) {
                p.setPen(QPen(Qt::white, 1));
                p.setBrush(Qt::white);
                p.drawEllipse(toScreen(pt.r * std::sin(pt.theta), pt.phi), 2, 2);
                p.setPen(QPen(ray.color, 1.2f));
                p.setBrush(Qt::NoBrush);
            }
            if (pt.r_turning) {
                // Cyan diamond for r-turning points
                QPointF sc = toScreen(pt.r * std::sin(pt.theta), pt.phi);
                p.setPen(QPen(QColor(0, 255, 255), 1));
                p.setBrush(QColor(0, 255, 255));
                const float d = 3.f;
                QPolygonF diamond;
                diamond << QPointF(sc.x(), sc.y()-d) << QPointF(sc.x()+d, sc.y())
                        << QPointF(sc.x(), sc.y()+d) << QPointF(sc.x()-d, sc.y());
                p.drawPolygon(diamond);
                p.setPen(QPen(ray.color, 1.2f));
                p.setBrush(Qt::NoBrush);
            }
        }
    }
    p.setClipping(false);

    // Legend (fixed position, not affected by zoom/pan)
    p.setFont(QFont("Monospace", 8));
    int ly = rc.top() + 12;
    auto txt = [&](const QString& s, QColor c = Qt::white) {
        p.setPen(c); p.drawText(rc.left() + 4, ly, s); ly += 13;
    };
    txt("上面図 (x-y 平面)  ρ=r sinθ");
    txt(QString("range: %1M  zoom: x%2")
        .arg(maxR / m_M, 0, 'f', 0).arg(m_zoom, 0, 'f', 2));
    txt("緑=ISCO  黄=光子球  暗赤=エルゴ  白●=θターン  シアン◇=rターン");
    txt("ホイール:ズーム  左ドラッグ:パン  ダブルクリック:リセット",
        QColor(180,180,180));
}

// ── Zoom / pan event handlers ─────────────────────────────────────────────────

void GeodesicView::wheelEvent(QWheelEvent* e) {
    if (!inOverhead(e->position().toPoint())) { e->ignore(); return; }

    float old_zoom = m_zoom;
    float factor   = (e->angleDelta().y() > 0) ? 1.25f : 1.0f / 1.25f;
    m_zoom = std::max(0.1f, std::min(200.f, m_zoom * factor));

    // Zoom centred on cursor: keep the world point under the cursor fixed.
    // Overhead panel centre in widget coords:
    float cx  = width()  / 6.f;     // center of left panel (first third)
    float cy  = height() * 0.50f;
    float mx  = (float)e->position().x();
    float my  = (float)e->position().y();
    float dx  = mx - cx;             // cursor relative to panel centre (screen)
    float dy  = my - cy;             // positive = down
    float ratio = m_zoom / old_zoom;
    // pan_new = cursor_offset*(1-ratio) + pan*ratio
    m_pan.setX(dx * (1.f - ratio) + m_pan.x() * ratio);
    m_pan.setY(dy * (1.f - ratio) + m_pan.y() * ratio);

    update();
    e->accept();
}

void GeodesicView::mousePressEvent(QMouseEvent* e) {
    if (e->button() == Qt::LeftButton && inOverhead(e->pos())) {
        m_dragging  = true;
        m_lastMouse = e->pos();
        e->accept();
    }
}

void GeodesicView::mouseMoveEvent(QMouseEvent* e) {
    if (m_dragging && (e->buttons() & Qt::LeftButton)) {
        QPoint delta = e->pos() - m_lastMouse;
        m_lastMouse  = e->pos();
        m_pan.setX(m_pan.x() + delta.x());
        m_pan.setY(m_pan.y() + delta.y());  // screen y: positive = down
        update();
        e->accept();
    }
}

void GeodesicView::mouseReleaseEvent(QMouseEvent* e) {
    if (e->button() == Qt::LeftButton) m_dragging = false;
}

void GeodesicView::mouseDoubleClickEvent(QMouseEvent* e) {
    if (inOverhead(e->pos())) {
        m_zoom = 1.f;
        m_pan  = {0.f, 0.f};
        update();
        e->accept();
    }
}

void GeodesicView::drawSideView(QPainter& p, const QRect& rc) {
    p.fillRect(rc, Qt::black);

    float maxR = 15.f * m_M;
    for (const auto& ray : m_rays)
        for (const auto& pt : ray.pts)
            maxR = std::max(maxR, std::min(pt.r, 300.f * m_M));
    maxR = std::min(maxR, 200.f * m_M);

    // BH at panel center; ρ=r sinθ extends right, z=r cosθ extends up
    float cx    = rc.left() + rc.width()  * 0.5f;
    float cy    = rc.top()  + rc.height() * 0.5f;
    float scale = std::min(rc.width(), rc.height()) * 0.43f / maxR;

    auto toSc = [&](float rho, float z) -> QPointF {
        return { cx + rho * scale, cy - z * scale };
    };

    // Rotation axis (ρ=0) – vertical dashed line
    p.setPen(QPen(QColor(60, 60, 60), 1, Qt::DashLine));
    p.drawLine((int)cx, rc.top(), (int)cx, rc.bottom());

    // Equatorial plane (z=0) – horizontal dashed line
    p.setPen(QPen(QColor(60, 60, 60), 1, Qt::DashLine));
    p.drawLine(rc.left(), (int)cy, rc.right(), (int)cy);

    // BH horizon – filled circle of radius r_h (sphere in 3D)
    float bh_px = m_r_h * scale;
    p.setPen(QPen(QColor(40, 40, 40), 2));
    p.setBrush(QColor(20, 20, 20));
    p.drawEllipse(QPointF(cx, cy), bh_px, bh_px);

    // Ergosphere (equatorial approximate sphere)
    float r_ergo = m_M + std::sqrt(std::max(m_M * m_M - 0.f, 0.f));
    if (r_ergo > m_r_h * 1.01f) {
        p.setPen(QPen(QColor(120, 0, 0), 1.5f));
        p.setBrush(Qt::NoBrush);
        p.drawEllipse(QPointF(cx, cy), r_ergo * scale, r_ergo * scale);
    }

    // Photon sphere(s)
    auto drawRingSide = [&](float r, QColor c) {
        p.setPen(QPen(c, 1.5f));
        p.setBrush(Qt::NoBrush);
        p.drawEllipse(QPointF(cx, cy), r * scale, r * scale);
    };
    if (std::abs(m_r_ph_hi - m_r_ph_lo) > 0.05f * m_M) {
        drawRingSide(m_r_ph_lo, QColor(240, 200, 0));
        drawRingSide(m_r_ph_hi, QColor(240, 200, 0));
    } else {
        drawRingSide(m_r_ph_lo, QColor(240, 200, 0));
    }

    // Camera dot
    float cam_rho = m_cam_r * std::sin(m_cam_theta);
    float cam_z   = m_cam_r * std::cos(m_cam_theta);
    p.setPen(QPen(Qt::white, 2));
    p.setBrush(Qt::white);
    p.drawEllipse(toSc(cam_rho, cam_z), 4, 4);

    // Geodesic paths in ρ-z plane
    p.setClipRect(rc);
    for (const auto& ray : m_rays) {
        if (ray.pts.empty()) continue;
        p.setPen(QPen(ray.color, 1.2f));
        p.setBrush(Qt::NoBrush);
        QPolygonF poly;
        for (const auto& pt : ray.pts) {
            float rho = pt.r * std::sin(pt.theta);
            float z   = pt.r * std::cos(pt.theta);
            poly << toSc(rho, z);
        }
        p.drawPolyline(poly);

        for (const auto& pt : ray.pts) {
            if (pt.turning) {
                float rho = pt.r * std::sin(pt.theta);
                float z   = pt.r * std::cos(pt.theta);
                p.setPen(QPen(Qt::white, 1));
                p.setBrush(Qt::white);
                p.drawEllipse(toSc(rho, z), 2, 2);
                p.setPen(QPen(ray.color, 1.2f));
                p.setBrush(Qt::NoBrush);
            }
            if (pt.r_turning) {
                float rho = pt.r * std::sin(pt.theta);
                float z   = pt.r * std::cos(pt.theta);
                QPointF sc = toSc(rho, z);
                p.setPen(QPen(QColor(0, 255, 255), 1));
                p.setBrush(QColor(0, 255, 255));
                const float d = 3.f;
                QPolygonF diamond;
                diamond << QPointF(sc.x(), sc.y()-d) << QPointF(sc.x()+d, sc.y())
                        << QPointF(sc.x(), sc.y()+d) << QPointF(sc.x()-d, sc.y());
                p.drawPolygon(diamond);
                p.setPen(QPen(ray.color, 1.2f));
                p.setBrush(Qt::NoBrush);
            }
        }
    }
    p.setClipping(false);

    // Legend
    p.setFont(QFont("Monospace", 8));
    int ly = rc.top() + 12;
    auto txt = [&](const QString& s, QColor c = Qt::white) {
        p.setPen(c); p.drawText(rc.left() + 4, ly, s); ly += 13;
    };
    txt("側面図 (ρ-z 平面)");
    txt("ρ=r sinθ  z=r cosθ");
    txt(QString("range: %1M").arg(maxR / m_M, 0, 'f', 0), QColor(180, 180, 180));
    txt("縦軸=北↑  白●=カメラ  白●=θターン  シアン◇=rターン", QColor(180, 180, 180));
}

void GeodesicView::drawThetaPlot(QPainter& p, const QRect& rc) {
    p.fillRect(rc, QColor(18, 18, 18));
    if (m_rays.empty()) return;

    const int LINE_H    = 13;
    const int N_RAYS    = (int)m_rays.size();
    const int LEGEND_H  = N_RAYS * LINE_H + 4;
    // exit_v strip: only shown when ≥2 escaped rays exist
    int nEscaped = 0;
    for (const auto& r : m_rays) if (r.escaped) nEscaped++;
    const int EXIT_H    = (nEscaped >= 2) ? std::max(50, rc.height() / 6) : 0;
    const int PLOT_H    = std::max(rc.height() - LEGEND_H - EXIT_H, 40);

    // ── Legend area (fixed strip at bottom) ──────────────────────────────────
    QRect legendRc(rc.left(), rc.bottom() - LEGEND_H, rc.width(), LEGEND_H);
    p.fillRect(legendRc, QColor(10, 10, 10));
    p.setPen(QColor(50, 50, 50));
    p.drawLine(legendRc.left(), legendRc.top(), legendRc.right(), legendRc.top());

    // Detect non-monotone exit_v for colour-coding in legend
    // Collect (scan_pos, ray_idx) for escaped rays in order
    std::vector<std::pair<float,int>> escIdx;
    for (int i = 0; i < N_RAYS; ++i)
        if (m_rays[i].escaped) {
            float sp = m_isVertical ? m_rays[i].ndc_y : m_rays[i].ndc_x;
            escIdx.push_back({sp, i});
        }
    std::sort(escIdx.begin(), escIdx.end(),
              [](auto& a, auto& b){ return a.first < b.first; });
    // 垂直スイープのみ単調性フラグを立てる (水平では exit_v の非単調は正常物理)
    // 閾値 0.003 でノイズ (~0.0001) を除外しフォトンリング境界を検出
    constexpr float MONO_THR = 0.003f;
    std::vector<bool> nonMono(N_RAYS, false);
    if (m_isVertical) {
        for (int k = 1; k < (int)escIdx.size(); ++k) {
            float dv = m_rays[escIdx[k].second].exit_v - m_rays[escIdx[k-1].second].exit_v;
            if (k >= 2) {
                float dv_prev = m_rays[escIdx[k-1].second].exit_v - m_rays[escIdx[k-2].second].exit_v;
                if (std::abs(dv) > MONO_THR && std::abs(dv_prev) > MONO_THR &&
                    ((dv > 0) != (dv_prev > 0))) {
                    nonMono[escIdx[k].second]   = true;
                    nonMono[escIdx[k-1].second] = true;
                }
            }
        }
    }

    p.setFont(QFont("Monospace", 8));
    int ly2 = legendRc.top() + LINE_H;
    const char* scanLabel = m_isVertical ? "ndc_y" : "ndc_x";
    for (int i = 0; i < N_RAYS; ++i) {
        bool badTheta = m_rays[i].Theta_min < -1e-6f;
        bool badMono  = nonMono[i];
        QColor col = (badTheta || badMono) ? QColor(255, 80, 80) : m_rays[i].color;
        p.setPen(col);
        float scanVal = m_isVertical ? m_rays[i].ndc_y : m_rays[i].ndc_x;
        p.drawText(rc.left() + 4, ly2,
                   QString("Ray%1: %2=%3  turns=%4  Θmin=%5  v=%6%7%8")
                   .arg(i+1)
                   .arg(scanLabel)
                   .arg(scanVal,              0, 'f', 3)
                   .arg(m_rays[i].n_theta_turns)
                   .arg(m_rays[i].Theta_min,  0, 'e', 2)
                   .arg(m_rays[i].escaped ? QString::number(m_rays[i].exit_v, 'f', 3) : QString("---"))
                   .arg(badTheta ? "  ⚠Θ" : "")
                   .arg(badMono  ? "  ⚠単調" : ""));
        ly2 += LINE_H;
    }

    // ── exit_v monotonicity strip ─────────────────────────────────────────────
    if (EXIT_H > 0) {
        QRect exitRc(rc.left(), rc.top() + PLOT_H, rc.width(), EXIT_H);
        p.fillRect(exitRc, QColor(12, 12, 25));
        p.setPen(QColor(50, 50, 80));
        p.drawLine(exitRc.left(), exitRc.top(), exitRc.right(), exitRc.top());

        // X axis = scan position (ndc_x or ndc_y), Y axis = exit_v [0,1]
        float xMin = -1.0f, xMax = 1.0f;
        for (const auto& r : m_rays) {
            float sp = m_isVertical ? r.ndc_y : r.ndc_x;
            xMin = std::min(xMin, sp - 0.05f);
            xMax = std::max(xMax, sp + 0.05f);
        }
        float xRange = std::max(xMax - xMin, 1e-4f);

        auto toExitSc = [&](float sp, float v) -> QPointF {
            float xf = (sp - xMin) / xRange;
            float yf = 1.0f - v; // v=0 (north pole) → top; v=1 (south pole) → bottom
            return { (float)exitRc.left() + xf * exitRc.width(),
                     (float)exitRc.top()  + yf * (exitRc.height() - 2) };
        };

        // Equatorial guide (v=0.5)
        float ymid = (float)exitRc.top() + 0.5f * (exitRc.height() - 2);
        p.setPen(QPen(QColor(80, 80, 80), 1, Qt::DashLine));
        p.drawLine(exitRc.left(), (int)ymid, exitRc.right(), (int)ymid);

        // Connect escaped rays sorted by scan position
        if (!escIdx.empty()) {
            QPolygonF poly;
            for (auto& [sp, idx] : escIdx)
                poly << toExitSc(sp, m_rays[idx].exit_v);
            p.setPen(QPen(QColor(200, 200, 60), 1));
            p.setBrush(Qt::NoBrush);
            p.drawPolyline(poly);
        }
        // Dots per ray
        for (const auto& r : m_rays) {
            if (!r.escaped) continue;
            float sp = m_isVertical ? r.ndc_y : r.ndc_x;
            p.setPen(Qt::NoPen);
            p.setBrush(nonMono[&r - &m_rays[0]] ? QColor(255,80,80) : r.color);
            p.drawEllipse(toExitSc(sp, r.exit_v), 3, 3);
        }

        p.setFont(QFont("Monospace", 7));
        p.setPen(QColor(160, 160, 100));
        p.drawText(exitRc.left() + 4, exitRc.top() + 10,
                   QString("exit_v vs %1  (北=上 0 / 赤道=中 0.5 / 南=下 1)  ⚠=単調性違反")
                   .arg(scanLabel));
    }

    // ── Plot area (curves only, clipped to upper region) ─────────────────────
    QRect plotRc(rc.left(), rc.top(), rc.width(), PLOT_H);
    p.setClipRect(plotRc);

    int   maxSteps = 0;
    float maxTh = 1.f, minTh = 0.f;
    for (const auto& ray : m_rays) {
        maxSteps = std::max(maxSteps, (int)ray.pts.size());
        for (const auto& pt : ray.pts) {
            maxTh = std::max(maxTh, pt.Theta_val);
            minTh = std::min(minTh, pt.Theta_val);
        }
    }
    if (maxSteps == 0) { p.setClipping(false); return; }
    minTh = std::min(minTh, -0.05f * maxTh);
    float yRange = maxTh - minTh;
    if (yRange < 1e-8f) yRange = 1.f;

    auto toSc = [&](int step, float tv) -> QPointF {
        float xf = (float)step / maxSteps;
        float yf = (tv - minTh) / yRange;
        return { (float)plotRc.left() + xf * plotRc.width(),
                 (float)plotRc.bottom() - yf * plotRc.height() };
    };

    // Zero line (Θ=0)
    float y0 = (float)plotRc.bottom() + minTh / yRange * plotRc.height();
    p.setPen(QPen(QColor(200, 60, 60), 1, Qt::DashLine));
    p.drawLine(plotRc.left(), (int)y0, plotRc.right(), (int)y0);
    p.setPen(QColor(200, 60, 60));
    p.setFont(QFont("Monospace", 8));
    p.drawText(plotRc.right() - 55, (int)y0 - 3, "Θ = 0");

    // Curves
    for (const auto& ray : m_rays) {
        if (ray.pts.empty()) continue;
        p.setPen(QPen(ray.color, 1.3f));
        QPolygonF poly;
        for (int s = 0; s < (int)ray.pts.size(); ++s)
            poly << toSc(s, ray.pts[s].Theta_val);
        p.drawPolyline(poly);

        for (int s = 0; s < (int)ray.pts.size(); ++s) {
            if (ray.pts[s].turning) {
                p.setPen(QPen(Qt::white, 1));
                p.setBrush(Qt::white);
                p.drawEllipse(toSc(s, ray.pts[s].Theta_val), 2, 2);
                p.setPen(QPen(ray.color, 1.3f));
                p.setBrush(Qt::NoBrush);
            }
        }
    }
    p.setClipping(false);

    // Title (overlay on top of plot)
    p.setPen(Qt::white);
    p.setFont(QFont("Monospace", 8));
    p.drawText(plotRc.left() + 4, plotRc.top() + 12, "Θ(θ) vs ステップ");
    p.drawText(plotRc.left() + 4, plotRc.top() + 24, "Θ<0 → パリティバグ");
}

// ── Geodesic3DView ────────────────────────────────────────────────────────────

Geodesic3DView::Geodesic3DView(QWidget* parent) : QWidget(parent) {
    setMinimumSize(400, 300);
}

void Geodesic3DView::setData(const std::vector<GeodesicRay>& rays,
                              float r_h, float r_ph_lo, float r_ph_hi,
                              float r_isco, float M,
                              float cam_r, float cam_theta, float cam_phi) {
    m_rays = rays;
    m_r_h = r_h; m_r_ph_lo = r_ph_lo; m_r_ph_hi = r_ph_hi;
    m_r_isco = r_isco; m_M = M;
    m_cam_r = cam_r; m_cam_theta = cam_theta; m_cam_phi = cam_phi;

    float maxR = 15.f * M;
    for (const auto& ray : rays)
        for (const auto& pt : ray.pts)
            maxR = std::max(maxR, std::min(pt.r, 300.f * M));
    m_dist = std::min(maxR, 200.f * M) * 3.5f;
    update();
}

// Perspective-project world point → screen.  Returns false if behind camera.
bool Geodesic3DView::projectPt(float wx, float wy, float wz, QPointF& out) const {
    float cosE = std::cos(m_elev), sinE = std::sin(m_elev);
    float cosA = std::cos(m_azi),  sinA = std::sin(m_azi);

    // Camera position
    float cx = m_dist * cosE * cosA;
    float cy = m_dist * cosE * sinA;
    float cz = m_dist * sinE;

    // Forward = unit vector from camera to origin
    float fx = -cosE*cosA, fy = -cosE*sinA, fz = -sinE;

    // Right = normalize(forward × world_z) = normalize((fy, -fx, 0))
    float rx, ry;
    if (cosE > 0.02f) { rx = fy / cosE; ry = -fx / cosE; }
    else              { rx = -sinA;     ry =  cosA;       }
    // rz = 0 by construction

    // Up = right × forward  (rz=0 simplifies two terms)
    float ux = ry * fz;
    float uy = -rx * fz;
    float uz = rx * fy - ry * fx;

    float dx = wx - cx, dy = wy - cy, dz = wz - cz;
    float vr  = dx*rx + dy*ry;               // dz*0 dropped
    float vu  = dx*ux + dy*uy + dz*uz;
    float vfd = dx*fx + dy*fy + dz*fz;

    if (vfd <= 0.01f) return false;
    float focal = std::min(width(), height()) * 0.55f;
    out.setX(width()  * 0.5f + vr * focal / vfd);
    out.setY(height() * 0.5f - vu * focal / vfd);
    return true;
}

// Draw a circle of radius r in the plane whose normal is (nx,ny,nz), centred at origin.
void Geodesic3DView::drawRing3D(QPainter& p, float r,
                                  float nx, float ny, float nz, QColor c, int seg) {
    float nlen = std::sqrt(nx*nx + ny*ny + nz*nz);
    if (nlen < 1e-8f) return;
    nx/=nlen; ny/=nlen; nz/=nlen;

    // Two unit vectors in the plane
    float u1x, u1y, u1z;
    if (std::abs(nz) < 0.99f) {
        float l = std::sqrt(nx*nx + ny*ny);
        u1x = -ny/l; u1y = nx/l; u1z = 0.f;
    } else {
        u1x = 1.f; u1y = 0.f; u1z = 0.f;
    }
    float u2x = ny*u1z - nz*u1y;
    float u2y = nz*u1x - nx*u1z;
    float u2z = nx*u1y - ny*u1x;

    constexpr float TWO_PI = 6.28318530718f;
    p.setPen(QPen(c, 1.5f));
    p.setBrush(Qt::NoBrush);
    QPolygonF poly;
    for (int i = 0; i <= seg; ++i) {
        float t = TWO_PI * i / seg;
        float ct = std::cos(t), st = std::sin(t);
        QPointF sc;
        if (projectPt(r*(ct*u1x+st*u2x), r*(ct*u1y+st*u2y), r*(ct*u1z+st*u2z), sc))
            poly << sc;
        else if (!poly.isEmpty()) { p.drawPolyline(poly); poly.clear(); }
    }
    if (!poly.isEmpty()) p.drawPolyline(poly);
}

// Draw a circle of radius r at height z (in the z=const plane).
void Geodesic3DView::drawCircleAtZ(QPainter& p, float r, float z, QColor c, int seg) {
    constexpr float TWO_PI = 6.28318530718f;
    p.setPen(QPen(c, 1.f));
    p.setBrush(Qt::NoBrush);
    QPolygonF poly;
    for (int i = 0; i <= seg; ++i) {
        float t = TWO_PI * i / seg;
        QPointF sc;
        if (projectPt(r*std::cos(t), r*std::sin(t), z, sc))
            poly << sc;
        else if (!poly.isEmpty()) { p.drawPolyline(poly); poly.clear(); }
    }
    if (!poly.isEmpty()) p.drawPolyline(poly);
}

void Geodesic3DView::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    p.fillRect(rect(), Qt::black);

    if (m_rays.empty()) {
        p.setPen(Qt::white);
        p.setFont(QFont("Monospace", 10));
        p.drawText(rect(), Qt::AlignCenter,
                   "トレース実行後に3Dビューが表示されます\n"
                   "左ドラッグ:回転  ホイール:ズーム  ダブルクリック:リセット");
        return;
    }

    float maxR = 15.f * m_M;
    for (const auto& ray : m_rays)
        for (const auto& pt : ray.pts)
            maxR = std::max(maxR, std::min(pt.r, 300.f * m_M));
    maxR = std::min(maxR, 200.f * m_M);

    // Equatorial reference grid
    for (float gr = 5.f * m_M; gr <= maxR * 1.1f; gr += 5.f * m_M)
        drawCircleAtZ(p, gr, 0.f, QColor(28, 28, 28), 32);

    // Coordinate axes
    {
        float axLen = maxR * 0.4f;
        p.setFont(QFont("Monospace", 9));
        QPointF ori;
        bool oriOk = projectPt(0, 0, 0, ori);
        struct { float x, y, z; QColor c; const char* l; } ax[] = {
            {axLen, 0, 0, {180,60,60}, "x"},
            {0, axLen, 0, {60,180,60}, "y"},
            {0, 0, axLen, {60,80,200}, "z(N)"},
        };
        for (auto& a3 : ax) {
            QPointF tip;
            if (oriOk && projectPt(a3.x, a3.y, a3.z, tip)) {
                p.setPen(QPen(a3.c, 1.5f));
                p.drawLine(ori, tip);
                p.setPen(a3.c);
                p.drawText(tip + QPointF(4, -2), a3.l);
            }
        }
    }

    // Event horizon wireframe
    {
        QColor bhWire(55, 55, 85);
        drawCircleAtZ(p, m_r_h, 0.f, bhWire, 32);
        drawRing3D(p, m_r_h, 0, 1, 0, bhWire, 32);   // xz-plane meridian
        drawRing3D(p, m_r_h, 1, 0, 0, bhWire, 32);   // yz-plane meridian
        constexpr float DEG2RAD = 3.14159265f / 180.f;
        for (float lat : {30.f, 60.f}) {
            float lr = lat * DEG2RAD;
            drawCircleAtZ(p, m_r_h*std::cos(lr),  m_r_h*std::sin(lr), bhWire, 24);
            drawCircleAtZ(p, m_r_h*std::cos(lr), -m_r_h*std::sin(lr), bhWire, 24);
        }
    }

    // ISCO (equatorial)
    drawCircleAtZ(p, m_r_isco, 0.f, QColor(0, 160, 70), 64);

    // Photon sphere ring(s)
    if (std::abs(m_r_ph_hi - m_r_ph_lo) > 0.05f * m_M) {
        drawCircleAtZ(p, m_r_ph_lo, 0.f, QColor(200, 170, 0), 64);
        drawCircleAtZ(p, m_r_ph_hi, 0.f, QColor(200, 170, 0), 64);
    } else {
        drawCircleAtZ(p, m_r_ph_lo, 0.f, QColor(200, 170, 0), 64);
    }

    // Simulated camera position
    {
        float wx = m_cam_r * std::sin(m_cam_theta) * std::cos(m_cam_phi);
        float wy = m_cam_r * std::sin(m_cam_theta) * std::sin(m_cam_phi);
        float wz = m_cam_r * std::cos(m_cam_theta);
        QPointF sc;
        if (projectPt(wx, wy, wz, sc)) {
            p.setPen(QPen(Qt::white, 2));
            p.setBrush(Qt::white);
            p.drawEllipse(sc, 4, 4);
        }
    }

    // Geodesic paths
    for (const auto& ray : m_rays) {
        if (ray.pts.empty()) continue;
        p.setPen(QPen(ray.color, 1.3f));
        p.setBrush(Qt::NoBrush);
        QPolygonF poly;
        for (const auto& pt : ray.pts) {
            float wx = pt.r * std::sin(pt.theta) * std::cos(pt.phi);
            float wy = pt.r * std::sin(pt.theta) * std::sin(pt.phi);
            float wz = pt.r * std::cos(pt.theta);
            QPointF sc;
            if (projectPt(wx, wy, wz, sc))
                poly << sc;
            else if (!poly.isEmpty()) { p.drawPolyline(poly); poly.clear(); }
        }
        if (!poly.isEmpty()) p.drawPolyline(poly);

        // θ-turning-point markers
        for (const auto& pt : ray.pts) {
            if (!pt.turning) continue;
            float wx = pt.r * std::sin(pt.theta) * std::cos(pt.phi);
            float wy = pt.r * std::sin(pt.theta) * std::sin(pt.phi);
            float wz = pt.r * std::cos(pt.theta);
            QPointF sc;
            if (projectPt(wx, wy, wz, sc)) {
                p.setPen(QPen(Qt::white, 1));
                p.setBrush(Qt::white);
                p.drawEllipse(sc, 2, 2);
                p.setPen(QPen(ray.color, 1.3f));
                p.setBrush(Qt::NoBrush);
            }
        }
    }

    // HUD legend
    p.setFont(QFont("Monospace", 8));
    {
        constexpr float RAD2DEG = 180.f / 3.14159265f;
        int ly = 14;
        auto lbl = [&](const QString& s, QColor c = Qt::white) {
            p.setPen(c); p.drawText(4, ly, s); ly += 13;
        };
        lbl("3D 測地線ビュー");
        lbl(QString("視点 azi=%1°  elev=%2°  |  z = スピン軸(北)")
            .arg(m_azi * RAD2DEG, 0,'f',0)
            .arg(m_elev * RAD2DEG, 0,'f',0), QColor(180,180,180));
        lbl("緑=ISCO  黄=光子球  白●=観測者カメラ", QColor(180,180,180));
        lbl("左ドラッグ:回転  ホイール:ズーム  ダブルクリック:リセット", QColor(140,140,140));
    }
}

void Geodesic3DView::wheelEvent(QWheelEvent* e) {
    float factor = (e->angleDelta().y() > 0) ? 0.87f : 1.0f / 0.87f;
    m_dist = std::max(m_M * 5.f, m_dist * factor);
    update();
    e->accept();
}

void Geodesic3DView::mousePressEvent(QMouseEvent* e) {
    if (e->button() == Qt::LeftButton) {
        m_dragging  = true;
        m_lastMouse = e->pos();
        e->accept();
    }
}

void Geodesic3DView::mouseMoveEvent(QMouseEvent* e) {
    if (m_dragging && (e->buttons() & Qt::LeftButton)) {
        QPoint d    = e->pos() - m_lastMouse;
        m_lastMouse = e->pos();
        m_azi  -= d.x() * 0.008f;
        m_elev += d.y() * 0.008f;
        constexpr float HALF_PI = 1.5707963f;
        m_elev = std::max(-HALF_PI * 0.98f, std::min(HALF_PI * 0.98f, m_elev));
        update();
        e->accept();
    }
}

void Geodesic3DView::mouseReleaseEvent(QMouseEvent* e) {
    if (e->button() == Qt::LeftButton) m_dragging = false;
}

void Geodesic3DView::mouseDoubleClickEvent(QMouseEvent* e) {
    m_azi  = 0.6f;
    m_elev = 0.4f;
    float maxR = 15.f * m_M;
    for (const auto& ray : m_rays)
        for (const auto& pt : ray.pts)
            maxR = std::max(maxR, std::min(pt.r, 300.f * m_M));
    m_dist = std::min(maxR, 200.f * m_M) * 3.5f;
    update();
    e->accept();
}

// ── GeodesicDebugWidget ───────────────────────────────────────────────────────

GeodesicDebugWidget::GeodesicDebugWidget(QWidget* parent) : QWidget(parent) {
    auto* vlay = new QVBoxLayout(this);
    vlay->setSpacing(4);

    // ── Row 1: trace button, ray count, debug step count ────────────────────
    auto* hlay = new QHBoxLayout;
    auto* btn  = new QPushButton("トレース実行");
    m_spinRays = new QSpinBox;
    m_spinRays->setRange(1, 200);
    m_spinRays->setValue(7);
    m_spinRays->setToolTip("スキャン本数（水平/垂直/直線モード）  グリッドモード時はNx×Nyを使用");
    m_spinDebugSteps = new QSpinBox;
    m_spinDebugSteps->setRange(100, 50000);
    m_spinDebugSteps->setValue(8000);
    m_spinDebugSteps->setSingleStep(500);
    m_spinDebugSteps->setToolTip("CPUデバッグ積分の最大ステップ数");
    hlay->addWidget(btn);
    hlay->addWidget(new QLabel("本数:"));
    hlay->addWidget(m_spinRays);
    hlay->addWidget(new QLabel("  ステップ上限:"));
    hlay->addWidget(m_spinDebugSteps);
    hlay->addStretch();
    vlay->addLayout(hlay);

    // ── Row 2: focus zone ────────────────────────────────────────────────────
    auto* flay = new QHBoxLayout;
    m_chkFocus = new QCheckBox("フォーカスゾーン");
    m_chkFocus->setToolTip("指定スキャン軸範囲を集中サンプリング（水平/垂直/直線モード）");
    m_spinFocusMin = new QDoubleSpinBox;
    m_spinFocusMin->setRange(-2.0, 2.0); m_spinFocusMin->setValue(-0.2);
    m_spinFocusMin->setSingleStep(0.05); m_spinFocusMin->setDecimals(2);
    m_spinFocusMax = new QDoubleSpinBox;
    m_spinFocusMax->setRange(-2.0, 2.0); m_spinFocusMax->setValue(0.2);
    m_spinFocusMax->setSingleStep(0.05); m_spinFocusMax->setDecimals(2);
    m_spinFocusRays = new QSpinBox;
    m_spinFocusRays->setRange(1, 200); m_spinFocusRays->setValue(30);
    m_spinFocusRays->setToolTip("フォーカスゾーン内の追加ray本数");
    flay->addWidget(m_chkFocus);
    flay->addWidget(new QLabel("  min:"));
    flay->addWidget(m_spinFocusMin);
    flay->addWidget(new QLabel("max:"));
    flay->addWidget(m_spinFocusMax);
    flay->addWidget(new QLabel("  本数:"));
    flay->addWidget(m_spinFocusRays);
    flay->addStretch();
    vlay->addLayout(flay);

    // ── Row 3a: sweep mode buttons ───────────────────────────────────────────
    auto* slay = new QHBoxLayout;
    m_radioHoriz = new QRadioButton("水平");
    m_radioVert  = new QRadioButton("垂直");
    m_radioLine  = new QRadioButton("直線");
    m_radioGrid  = new QRadioButton("グリッド");
    m_radioHoriz->setChecked(true);
    auto* sweepGrp = new QButtonGroup(this);
    for (auto* r : {m_radioHoriz, m_radioVert, m_radioLine, m_radioGrid})
        sweepGrp->addButton(r);
    slay->addWidget(new QLabel("スイープ:"));
    slay->addWidget(m_radioHoriz);
    slay->addWidget(m_radioVert);
    slay->addWidget(m_radioLine);
    slay->addWidget(m_radioGrid);
    slay->addStretch();
    vlay->addLayout(slay);

    // ── 1D sweep panel (水平 / 垂直) ─────────────────────────────────────────
    m_sweep1DPanel = new QWidget;
    {
        auto* lay = new QHBoxLayout(m_sweep1DPanel);
        lay->setContentsMargins(0, 0, 0, 0);
        m_spinScanMin = new QDoubleSpinBox;
        m_spinScanMin->setRange(-2.0, 2.0); m_spinScanMin->setValue(-0.8);
        m_spinScanMin->setSingleStep(0.05); m_spinScanMin->setDecimals(2);
        m_spinScanMin->setToolTip("スキャン軸の開始 NDC 値");
        m_spinScanMax = new QDoubleSpinBox;
        m_spinScanMax->setRange(-2.0, 2.0); m_spinScanMax->setValue(0.8);
        m_spinScanMax->setSingleStep(0.05); m_spinScanMax->setDecimals(2);
        m_spinScanMax->setToolTip("スキャン軸の終了 NDC 値");
        m_spinFixedNdc = new QDoubleSpinBox;
        m_spinFixedNdc->setRange(-2.0, 2.0); m_spinFixedNdc->setValue(0.0);
        m_spinFixedNdc->setSingleStep(0.05); m_spinFixedNdc->setDecimals(3);
        m_spinFixedNdc->setToolTip("固定する垂直方向の NDC 値");
        lay->addWidget(new QLabel("  スキャン範囲:"));
        lay->addWidget(m_spinScanMin);
        lay->addWidget(new QLabel("→"));
        lay->addWidget(m_spinScanMax);
        lay->addWidget(new QLabel("  固定値:"));
        lay->addWidget(m_spinFixedNdc);
        lay->addStretch();
    }
    vlay->addWidget(m_sweep1DPanel);

    // ── Arbitrary-line sweep panel ────────────────────────────────────────────
    m_sweepLinePanel = new QWidget;
    {
        auto* lay = new QHBoxLayout(m_sweepLinePanel);
        lay->setContentsMargins(0, 0, 0, 0);
        auto mkDS = [](double v) {
            auto* s = new QDoubleSpinBox;
            s->setRange(-2.0, 2.0); s->setValue(v);
            s->setSingleStep(0.05); s->setDecimals(3);
            return s;
        };
        m_spinLineX1 = mkDS(-0.8); m_spinLineY1 = mkDS(-0.8);
        m_spinLineX2 = mkDS( 0.8); m_spinLineY2 = mkDS( 0.8);
        lay->addWidget(new QLabel("  始点 (x₁,y₁):"));
        lay->addWidget(m_spinLineX1); lay->addWidget(m_spinLineY1);
        lay->addWidget(new QLabel("  終点 (x₂,y₂):"));
        lay->addWidget(m_spinLineX2); lay->addWidget(m_spinLineY2);
        lay->addStretch();
    }
    m_sweepLinePanel->setVisible(false);
    vlay->addWidget(m_sweepLinePanel);

    // ── 2D grid sweep panel ───────────────────────────────────────────────────
    m_sweepGridPanel = new QWidget;
    {
        auto* lay = new QHBoxLayout(m_sweepGridPanel);
        lay->setContentsMargins(0, 0, 0, 0);
        auto mkDS = [](double v) {
            auto* s = new QDoubleSpinBox;
            s->setRange(-2.0, 2.0); s->setValue(v);
            s->setSingleStep(0.05); s->setDecimals(2);
            return s;
        };
        m_spinGridXMin = mkDS(-0.8); m_spinGridXMax = mkDS(0.8);
        m_spinGridYMin = mkDS(-0.8); m_spinGridYMax = mkDS(0.8);
        m_spinGridNx = new QSpinBox; m_spinGridNx->setRange(1, 30); m_spinGridNx->setValue(5);
        m_spinGridNy = new QSpinBox; m_spinGridNy->setRange(1, 30); m_spinGridNy->setValue(5);
        lay->addWidget(new QLabel("  x:"));
        lay->addWidget(m_spinGridXMin); lay->addWidget(new QLabel("→")); lay->addWidget(m_spinGridXMax);
        lay->addWidget(new QLabel("  y:"));
        lay->addWidget(m_spinGridYMin); lay->addWidget(new QLabel("→")); lay->addWidget(m_spinGridYMax);
        lay->addWidget(new QLabel("  Nx:")); lay->addWidget(m_spinGridNx);
        lay->addWidget(new QLabel("Ny:"));  lay->addWidget(m_spinGridNy);
        lay->addStretch();
    }
    m_sweepGridPanel->setVisible(false);
    vlay->addWidget(m_sweepGridPanel);

    // Connect mode buttons → panel visibility
    auto updatePanels = [this]() {
        bool is1D = m_radioHoriz->isChecked() || m_radioVert->isChecked();
        m_sweep1DPanel->setVisible(is1D);
        m_sweepLinePanel->setVisible(m_radioLine->isChecked());
        m_sweepGridPanel->setVisible(m_radioGrid->isChecked());
        m_spinRays->setEnabled(!m_radioGrid->isChecked());
    };
    for (auto* r : {m_radioHoriz, m_radioVert, m_radioLine, m_radioGrid})
        connect(r, &QRadioButton::toggled, this, updatePanels);

    // ── Row 4: winding limit ─────────────────────────────────────────────────
    auto* wlay = new QHBoxLayout;
    m_spinMaxOrbits = new QSpinBox;
    m_spinMaxOrbits->setRange(1, 200);
    m_spinMaxOrbits->setValue(8);
    m_spinMaxOrbits->setToolTip(
        "光子が最大何周（2π rad）回ったら強制終了するか。"
        "0 = 無制限（注意：光子球近くは非常に遅い）");
    wlay->addWidget(new QLabel("最大周回数（φ winding）:"));
    wlay->addWidget(m_spinMaxOrbits);
    wlay->addWidget(new QLabel("周  ※光子球近くに設定して収束確認"));
    wlay->addStretch();
    vlay->addLayout(wlay);

    // Main splitter: view tabs (top) / log (bottom)
    auto* splitter = new QSplitter(Qt::Vertical);
    auto* viewTabs = new QTabWidget;
    m_view   = new GeodesicView;
    m_view3d = new Geodesic3DView;
    viewTabs->addTab(m_view,   "2Dビュー");
    viewTabs->addTab(m_view3d, "3Dビュー");
    splitter->addWidget(viewTabs);

    m_log = new QTextEdit;
    m_log->setReadOnly(true);
    m_log->setFont(QFont("Monospace", 8));
    m_log->setMaximumHeight(200);
    splitter->addWidget(m_log);
    splitter->setStretchFactor(0, 3);
    splitter->setStretchFactor(1, 1);
    vlay->addWidget(splitter);

    connect(btn, &QPushButton::clicked, this, &GeodesicDebugWidget::trace);
}

void GeodesicDebugWidget::setParams(const RenderParams* p) {
    m_params = p;
}

void GeodesicDebugWidget::appendLog(const QString& s) {
    m_log->append(s);
    qDebug().noquote() << s;   // also mirror to main console / Qt output
}

void GeodesicDebugWidget::trace() {
    if (!m_params) return;
    const RenderParams& p = *m_params;
    float M = p.bh.M, a = p.bh.a, Q = p.bh.Q;
    float aspect    = p.width > 0 ? (float)p.width / p.height : 1.f;
    float tan_hfov  = std::tan(p.fov * 0.5f);
    float st = std::sin(p.cam_theta), ct = std::cos(p.cam_theta);
    float sp = std::sin(p.cam_phi),   cp = std::cos(p.cam_phi);

    int N = m_spinRays->value();
    int cpu_steps = m_spinDebugSteps->value();
    std::vector<GeodesicRay> rays;
    m_log->clear();

    appendLog("=== 測地線トレース ===");
    appendLog(QString("M=%1  a/M=%2  Q/M=%3  cam_r=%4  θ=%5  φ=%6")
              .arg(M,         0,'f',3)
              .arg(a/M,       0,'f',3)
              .arg(Q/M,       0,'f',3)
              .arg(p.cam_r,   0,'f',3)
              .arg(p.cam_theta,0,'f',4)
              .arg(p.cam_phi, 0,'f',4));
    constexpr float TWO_PI = 6.28318530718f;
    float max_phi_wind = m_spinMaxOrbits->value() * TWO_PI;

    appendLog(QString("step_size=%1  max_steps=%2  debug_steps=%3  max_orbits=%4")
              .arg(p.step_size, 0, 'f', 4).arg(p.max_steps).arg(cpu_steps)
              .arg(m_spinMaxOrbits->value()));

    bool isHoriz    = m_radioHoriz->isChecked();
    bool isVert     = m_radioVert->isChecked();
    bool isLine     = m_radioLine->isChecked();
    bool isGrid     = m_radioGrid->isChecked();
    bool isVertical = isVert; // used by 2D-view monotonicity check

    // SampleSpec carries total so colour palette scales correctly per-mode
    struct SampleSpec { float ndc_x; float ndc_y; bool focus; int idx; int total; };
    std::vector<SampleSpec> samples;

    if (isGrid) {
        float xmin=(float)m_spinGridXMin->value(), xmax=(float)m_spinGridXMax->value();
        float ymin=(float)m_spinGridYMin->value(), ymax=(float)m_spinGridYMax->value();
        int nx=m_spinGridNx->value(), ny=m_spinGridNy->value();
        for (int iy=0; iy<ny; ++iy)
            for (int ix=0; ix<nx; ++ix) {
                float x=(nx>1) ? xmin+(xmax-xmin)*ix/(nx-1) : (xmin+xmax)*0.5f;
                float y=(ny>1) ? ymin+(ymax-ymin)*iy/(ny-1) : (ymin+ymax)*0.5f;
                samples.push_back({x, y, false, iy*nx+ix, nx*ny});
            }
        appendLog(QString("グリッドスイープ: x=[%1,%2] y=[%3,%4]  %5×%6=%7本")
                  .arg(xmin,0,'f',2).arg(xmax,0,'f',2)
                  .arg(ymin,0,'f',2).arg(ymax,0,'f',2)
                  .arg(nx).arg(ny).arg(nx*ny));
    } else if (isLine) {
        float x1=(float)m_spinLineX1->value(), y1=(float)m_spinLineY1->value();
        float x2=(float)m_spinLineX2->value(), y2=(float)m_spinLineY2->value();
        for (int i=0; i<N; ++i) {
            float t=(N>1) ? (float)i/(N-1) : 0.5f;
            samples.push_back({x1+(x2-x1)*t, y1+(y2-y1)*t, false, i, N});
        }
        appendLog(QString("直線スイープ: (%1,%2)→(%3,%4)  %5本")
                  .arg(x1,0,'f',3).arg(y1,0,'f',3)
                  .arg(x2,0,'f',3).arg(y2,0,'f',3).arg(N));
    } else {
        // 1-D sweep (horizontal or vertical)
        float scanMin  = (float)m_spinScanMin->value();
        float scanMax  = (float)m_spinScanMax->value();
        float fixedNdc = (float)m_spinFixedNdc->value();
        for (int i=0; i<N; ++i) {
            float sv=(N>1) ? scanMin+(scanMax-scanMin)*i/(N-1) : (scanMin+scanMax)*0.5f;
            if (isVert) samples.push_back({fixedNdc, sv, false, i, N});
            else        samples.push_back({sv, fixedNdc, false, i, N});
        }
        appendLog(QString("スイープ方向: %1  範囲: [%2,%3]  固定値: %4=%5")
                  .arg(isVert ? "垂直 (ndc_y 変化)" : "水平 (ndc_x 変化)")
                  .arg(scanMin,0,'f',3).arg(scanMax,0,'f',3)
                  .arg(isVert ? "ndc_x" : "ndc_y")
                  .arg(fixedNdc,0,'f',3));
        if (m_chkFocus->isChecked()) {
            float fmin=(float)m_spinFocusMin->value();
            float fmax=(float)m_spinFocusMax->value();
            if (fmin>fmax) std::swap(fmin,fmax);
            int NF=m_spinFocusRays->value();
            const char* focusLabel = isVert ? "ndc_y" : "ndc_x";
            for (int i=0; i<NF; ++i) {
                float sv=(NF>1) ? fmin+(fmax-fmin)*i/(NF-1) : (fmin+fmax)*0.5f;
                if (isVert) samples.push_back({fixedNdc, sv, true, i, NF});
                else        samples.push_back({sv, fixedNdc, true, i, NF});
            }
            appendLog(QString("フォーカスゾーン: %1=[%2,%3]  %4本")
                      .arg(focusLabel)
                      .arg(fmin,0,'f',2).arg(fmax,0,'f',2).arg(NF));
        }
    }

    int totalRays = (int)samples.size();
    for (int si = 0; si < totalRays; ++si) {
        float ndc_x      = samples[si].ndc_x;
        float ndc_y      = samples[si].ndc_y;
        bool  isFocus    = samples[si].focus;
        int   colorIdx   = samples[si].idx;
        int   colorTotal = samples[si].total;

        float wdx, wdy, wdz;
        buildRayDir_cpu(ndc_x, ndc_y, aspect, tan_hfov, p, wdx, wdy, wdz);

        float n_r  =  st*cp*wdx + st*sp*wdy + ct*wdz;
        float n_th =  ct*cp*wdx + ct*sp*wdy - st*wdz;
        float n_ph = -sp*wdx    + cp*wdy;

        // ── Initial conditions ───────────────────────────────────────────────
        float b, q2, sr, sth;
        kn_initial_conditions_cpu(n_r, n_th, n_ph, p.cam_r, st, ct, M, a, Q,
                                   b, q2, sr, sth);

        float R0_init, Th0_init;
        KN::computeGeodesicRHS(p.cam_r, p.cam_theta, M, a, Q,
                                b, q2, R0_init, Th0_init);

        // Focus rays: white-ish palette; global rays: hue sweep
        QColor col;
        if (isFocus) {
            col = QColor::fromHsvF(colorTotal > 1 ? (float)colorIdx / (colorTotal - 1) : 0.f,
                                   0.3f, 1.0f);
        } else {
            col = QColor::fromHsvF(colorTotal > 1 ? (float)colorIdx / (colorTotal - 1) : 0.f,
                                   0.85f, 0.95f);
        }

        GeodesicRay ray = integrateCPU(b, q2, sr, sth,
                                        p.cam_r, p.cam_theta, p.cam_phi,
                                        M, a, Q, cpu_steps, p.step_size,
                                        max_phi_wind, p.sky_lon_offset, col, ndc_x);
        ray.ndc_y = ndc_y;

        appendLog(QString("\n-- Ray %1  ndc_x=%2  ndc_y=%3%4 --")
                  .arg(si + 1).arg(ndc_x, 0, 'f', 3).arg(ndc_y, 0, 'f', 3)
                  .arg(isFocus ? " [focus]" : ""));
        appendLog(QString("   n_r=%1  n_θ=%2  n_φ=%3")
                  .arg(n_r,  0, 'f', 5).arg(n_th, 0, 'f', 5).arg(n_ph, 0, 'f', 5));
        appendLog(QString("   R(r₀)=%1  Θ(θ₀)=%2%3%4")
                  .arg(R0_init, 0, 'e', 4).arg(Th0_init, 0, 'e', 4)
                  .arg(R0_init  < -1e-6f ? "  ⚠R<0" : "")
                  .arg(Th0_init < -1e-6f ? "  ⚠Θ<0" : ""));
        appendLog(QString("   b=%1  q²=%2  sr0=%3  sth0=%4")
                  .arg(ray.b, 0, 'f', 4).arg(ray.q2, 0, 'f', 4)
                  .arg(ray.pts.empty() ? 0 : ray.pts.front().sr_sign)
                  .arg(ray.pts.empty() ? 0 : ray.pts.front().sth_sign));
        appendLog(QString("   points=%1  θ-turns=%2  r-turns=%3  r_min=%4  φ_total=%5π")
                  .arg((int)ray.pts.size()).arg(ray.n_theta_turns).arg(ray.n_r_turns)
                  .arg(ray.r_min_turning < 9e8f ? ray.r_min_turning : 0.f, 0, 'f', 3)
                  .arg(ray.phi_total / 3.14159265f, 0, 'f', 2));
        appendLog(QString("   Θ_min=%1  null_max=%2  exit_uv=(%3, %4)")
                  .arg(ray.Theta_min, 0, 'e', 4).arg(ray.null_max, 0, 'e', 4)
                  .arg(ray.exit_u, 0, 'f', 4).arg(ray.exit_v, 0, 'f', 4));
        if (ray.escaped)
            appendLog(QString("   exit_dir=(%1,%2,%3)  rho_xy=%4  atan2=%5rad  [seam問題: |atan2|→π で u→0/1]")
                      .arg(ray.exit_ex,     0,'f',4)
                      .arg(ray.exit_ey,     0,'f',4)
                      .arg(ray.exit_ez,     0,'f',4)
                      .arg(ray.exit_rho_xy, 0,'f',5)
                      .arg(ray.exit_atan2,  0,'f',4));
        if (ray.absorbed) {
            appendLog("   → 吸収（地平線落下）");
        } else if (ray.winding_terminated) {
            appendLog(QString("   → ⏱ φ周回上限で強制終了 (%1周)")
                      .arg(m_spinMaxOrbits->value()));
        } else if (!ray.escaped) {
            appendLog("   → ⚠ max_steps到達（未収束）sky UV無効");
        } else {
            QString hemi = (ray.exit_v < 0.5f) ? "北半球" : "南半球";
            appendLog(QString("   → 脱出  sky UV=(%1, %2)  %3")
                      .arg(ray.exit_u, 0, 'f', 4).arg(ray.exit_v, 0, 'f', 4).arg(hemi));
        }
        if (ray.Theta_min < -1e-6f)
            appendLog(QString("   ⚠ Θ<0 検出 Θmin=%1 ← パリティバグの可能性")
                      .arg(ray.Theta_min, 0, 'e', 4));

        // turning points detail — suppress individual lines when there are too many
        if (ray.n_theta_turns + ray.n_r_turns <= 30) {
            for (int s = 0; s < (int)ray.pts.size(); ++s) {
                if (ray.pts[s].turning)
                    appendLog(QString("     ↺θ step%1: r=%2  θ=%3  Θ=%4  sr=%5")
                              .arg(s)
                              .arg(ray.pts[s].r,        0, 'f', 4)
                              .arg(ray.pts[s].theta,    0, 'f', 5)
                              .arg(ray.pts[s].Theta_val,0, 'e', 3)
                              .arg(ray.pts[s].sr_sign));
                if (ray.pts[s].r_turning)
                    appendLog(QString("     ↺r step%1: r=%2  θ=%3  R=%4  sth=%5")
                              .arg(s)
                              .arg(ray.pts[s].r,       0, 'f', 4)
                              .arg(ray.pts[s].theta,   0, 'f', 5)
                              .arg(ray.pts[s].R_val,   0, 'e', 3)
                              .arg(ray.pts[s].sth_sign));
            }
        } else {
            // Too many turns to list individually — show first 5 and last 5
            int shown = 0;
            bool gap_logged = false;
            for (int s = 0; s < (int)ray.pts.size(); ++s) {
                bool is_turn = ray.pts[s].turning || ray.pts[s].r_turning;
                if (!is_turn) continue;
                if (shown < 5) {
                    if (ray.pts[s].turning)
                        appendLog(QString("     ↺θ step%1: r=%2  θ=%3  Θ=%4")
                                  .arg(s).arg(ray.pts[s].r,0,'f',4)
                                  .arg(ray.pts[s].theta,0,'f',5)
                                  .arg(ray.pts[s].Theta_val,0,'e',3));
                    if (ray.pts[s].r_turning)
                        appendLog(QString("     ↺r step%1: r=%2  θ=%3  R=%4")
                                  .arg(s).arg(ray.pts[s].r,0,'f',4)
                                  .arg(ray.pts[s].theta,0,'f',5)
                                  .arg(ray.pts[s].R_val,0,'e',3));
                    shown++;
                } else if ((int)ray.pts.size() - s <= 5) {
                    if (!gap_logged) {
                        appendLog(QString("     ... (%1 turning points省略) ...")
                                  .arg(ray.n_theta_turns + ray.n_r_turns - 10));
                        gap_logged = true;
                    }
                    if (ray.pts[s].turning)
                        appendLog(QString("     ↺θ step%1: r=%2  θ=%3  Θ=%4")
                                  .arg(s).arg(ray.pts[s].r,0,'f',4)
                                  .arg(ray.pts[s].theta,0,'f',5)
                                  .arg(ray.pts[s].Theta_val,0,'e',3));
                    if (ray.pts[s].r_turning)
                        appendLog(QString("     ↺r step%1: r=%2  θ=%3  R=%4")
                                  .arg(s).arg(ray.pts[s].r,0,'f',4)
                                  .arg(ray.pts[s].theta,0,'f',5)
                                  .arg(ray.pts[s].R_val,0,'e',3));
                }
            }
        }
        rays.push_back(std::move(ray));
    } // end for samples

    // ── Monotonicity check for exit_v ────────────────────────────────────────
    appendLog("\n=== 完了 ===");
    {
        // Collect escaped rays sorted by scan coordinate
        struct ESc { float scanPos; float exit_v; int rayIdx; };
        std::vector<ESc> esc;
        for (int i = 0; i < (int)rays.size(); ++i) {
            if (!rays[i].escaped) continue;
            float sp = isVertical ? rays[i].ndc_y : rays[i].ndc_x;
            esc.push_back({sp, rays[i].exit_v, i});
        }
        std::sort(esc.begin(), esc.end(), [](auto& a, auto& b){ return a.scanPos < b.scanPos; });

        if (!isVertical) {
            // 水平スイープでは exit_v の単調性は物理的に要求されない
            // (異なる ndc_x は異なる天球緯度に到達するため非単調は正常)
            // 代わりに exit_v の分布範囲を報告する
            if (!esc.empty()) {
                float vmin = esc[0].exit_v, vmax = esc[0].exit_v;
                for (auto& e : esc) { vmin = std::min(vmin, e.exit_v); vmax = std::max(vmax, e.exit_v); }
                appendLog(QString("  ℹ 水平スイープ: exit_v 範囲 = [%1, %2]（単調性チェックは垂直スイープのみ）")
                          .arg(vmin, 0,'f',4).arg(vmax, 0,'f',4));
            }
        } else if ((int)esc.size() >= 2) {
            // 垂直スイープ: exit_v は単調であるべき
            // ノイズ閾値 0.003 (RK4 数値誤差 ~0.0001 の約30倍) で判定
            // フォトンリング付近 (b_eff ≈ b_crit) では非単調は物理的に起こり得る
            const float MONO_THR = 0.003f;
            const char* lbl = "ndc_y";
            int violations = 0;
            float dir0 = esc[1].exit_v - esc[0].exit_v;
            for (int k = 1; k < (int)esc.size(); ++k) {
                float dv = esc[k].exit_v - esc[k-1].exit_v;
                if (std::abs(dv) > MONO_THR && std::abs(dir0) > MONO_THR &&
                    (dv > 0.f) != (dir0 > 0.f)) {
                    appendLog(QString("  ⚠ 単調性違反 %1=%2→%3: exit_v=%4→%5 (Δv=%6)")
                              .arg(lbl)
                              .arg(esc[k-1].scanPos, 0,'f',3)
                              .arg(esc[k].scanPos,   0,'f',3)
                              .arg(esc[k-1].exit_v,  0,'f',4)
                              .arg(esc[k].exit_v,    0,'f',4)
                              .arg(std::abs(dv),     0,'f',4));
                    violations++;
                }
            }
            if (violations == 0)
                appendLog(QString("  ✓ exit_v 単調（垂直スイープ %1本の脱出レイ）")
                          .arg((int)esc.size()));
            else
                appendLog(QString("  ⚠ exit_v 非単調: %1箇所の違反（閾値 %2）→ フォトンリング or 3D測地線バグの可能性")
                          .arg(violations).arg(MONO_THR, 0,'f',3));
        } else {
            appendLog("  (脱出レイが少なすぎるため単調性チェックをスキップ)");
        }
    }

    // Photon sphere radii
    float r_ph_lo, r_ph_hi;
    if (std::abs(a) < 1e-5f * std::max(M, 1e-10f)) {
        r_ph_lo = r_ph_hi = 3.f * M;
    } else {
        float an = std::abs(a) / std::max(M, 1e-10f);
        r_ph_lo = 2.f*M*(1.f + std::cos(2.f/3.f * std::acos(-an)));
        r_ph_hi = 2.f*M*(1.f + std::cos(2.f/3.f * std::acos(+an)));
        if (r_ph_lo > r_ph_hi) std::swap(r_ph_lo, r_ph_hi);
    }

    {
        float r_h    = KN::r_plus(M, a, Q);
        float r_isco = KN::r_isco(M, a);
        m_view->setData(rays, r_h, r_ph_lo, r_ph_hi, r_isco, M,
                        p.cam_r, p.cam_theta, p.cam_phi, isVertical);
        m_view3d->setData(rays, r_h, r_ph_lo, r_ph_hi, r_isco, M,
                          p.cam_r, p.cam_theta, p.cam_phi);
    }
}
