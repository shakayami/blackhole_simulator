#pragma once
#include <cmath>

struct Camera {
    float r          = 20.0f;   // distance from BH centre (in M units)
    float theta      = 1.2f;    // polar angle [0, pi]
    float phi        = 0.0f;    // azimuthal angle
    float up_phi     = 0.0f;    // camera roll
    float fov        = 1.0f;    // horizontal FoV in radians (~57 deg)
    float look_yaw   = 0.0f;    // free-look: horizontal offset from BH-pointing baseline
    float look_pitch = 0.0f;    // free-look: vertical offset

    // Event-horizon radius — kept in sync with BH params by BlackHoleWidget.
    // zoom() scales the gap (r − r_horizon) so the camera approaches
    // asymptotically and never crosses the horizon.
    float r_horizon  = 2.0f;

    void orbit(float dTheta, float dPhi) {
        theta += dTheta;
        phi   += dPhi;
        theta  = std::max(0.05f, std::min(3.09f, theta));
    }

    void pan(float dYaw, float dPitch) {
        look_yaw   += dYaw;
        look_pitch += dPitch;
        look_pitch  = std::max(-(float)M_PI * 0.49f,
                               std::min( (float)M_PI * 0.49f, look_pitch));
    }

    // Zoom: scales the gap (r − r_horizon) by factor each scroll, so the camera
    // approaches the event horizon asymptotically and can NEVER cross it no matter
    // how many times the user scrolls in.
    void zoom(float factor) {
        float gap = r - r_horizon;
        if (gap <= 0.0f) gap = gapMin();   // recover if somehow inside
        gap *= factor;
        if (gap < gapMin()) gap = gapMin();
        r = r_horizon + gap;
        if (r > 2000.0f) r = 2000.0f;
    }

    // Call after changing r_horizon so the camera stays outside.
    void clampToHorizon() {
        if (r < r_horizon + gapMin()) r = r_horizon + gapMin();
    }

private:
    // Minimum allowed gap above the horizon: 1 ppm of r_horizon (absolute floor 1e-7).
    // This lets the camera get arbitrarily close while guaranteeing r > r_horizon always.
    float gapMin() const {
        float g = r_horizon * 1e-6f;
        return g > 1e-7f ? g : 1e-7f;
    }
public:

    void roll(float dRoll) { up_phi += dRoll; }
};
