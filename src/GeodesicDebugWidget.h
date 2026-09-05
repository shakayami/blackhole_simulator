#pragma once
#include <QWidget>
#include <QTextEdit>
#include <QCheckBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QRadioButton>
#include <QButtonGroup>
#include <QPointF>
#include <QMouseEvent>
#include <QWheelEvent>
#include <vector>
#include "cuda/RayTracer.cuh"

struct GeodesicPt {
    float r, theta, phi;
    float Theta_val;   // Theta(θ): always >=0 if parity correct
    float R_val;
    float null_resid;  // |g^μν p_μ p_ν| — should be ~0
    int   sth_sign;
    int   sr_sign;     // radial sign: +1=outgoing, -1=infalling
    bool  turning;     // theta turning point flagged here
    bool  r_turning;   // r turning point (R(r) went negative)
};

struct GeodesicRay {
    std::vector<GeodesicPt> pts;
    float  b, q2;
    bool   absorbed;
    bool   winding_terminated;
    bool   escaped;
    float  phi_total;
    float  exit_u, exit_v;
    float  exit_ex, exit_ey, exit_ez;
    float  exit_rho_xy;
    float  exit_atan2;
    int    n_theta_turns;
    int    n_r_turns;
    float  Theta_min;
    float  r_min_turning;
    float  null_max;
    QColor color;
    float  ndc_x;
    float  ndc_y = 0.f;
};

// Three-panel 2D drawing widget: overhead / side / Theta plot
class GeodesicView : public QWidget {
    Q_OBJECT
public:
    explicit GeodesicView(QWidget* parent = nullptr);
    void setData(const std::vector<GeodesicRay>& rays,
                 float r_h, float r_ph_lo, float r_ph_hi,
                 float r_isco, float M,
                 float cam_r, float cam_theta, float cam_phi,
                 bool isVertical = false);
protected:
    void paintEvent(QPaintEvent*) override;
    void wheelEvent(QWheelEvent*) override;
    void mousePressEvent(QMouseEvent*) override;
    void mouseMoveEvent(QMouseEvent*) override;
    void mouseReleaseEvent(QMouseEvent*) override;
    void mouseDoubleClickEvent(QMouseEvent*) override;
private:
    void drawOverhead(QPainter& p, const QRect& rc);
    void drawSideView(QPainter& p, const QRect& rc);
    void drawThetaPlot(QPainter& p, const QRect& rc);
    bool inOverhead(QPoint pos) const { return pos.x() < width() / 3; }

    std::vector<GeodesicRay> m_rays;
    float m_r_h = 2.f, m_r_ph_lo = 3.f, m_r_ph_hi = 3.f;
    float m_r_isco = 6.f, m_M = 1.f;
    float m_cam_r = 20.f, m_cam_theta = 1.5707f, m_cam_phi = 0.f;
    bool  m_isVertical = false;

    float   m_zoom = 1.f;
    QPointF m_pan  = {0.f, 0.f};
    QPoint  m_lastMouse;
    bool    m_dragging = false;
};

// Interactive 3D perspective view of geodesics (software rasteriser)
class Geodesic3DView : public QWidget {
    Q_OBJECT
public:
    explicit Geodesic3DView(QWidget* parent = nullptr);
    void setData(const std::vector<GeodesicRay>& rays,
                 float r_h, float r_ph_lo, float r_ph_hi,
                 float r_isco, float M,
                 float cam_r, float cam_theta, float cam_phi);
protected:
    void paintEvent(QPaintEvent*) override;
    void wheelEvent(QWheelEvent*) override;
    void mousePressEvent(QMouseEvent*) override;
    void mouseMoveEvent(QMouseEvent*) override;
    void mouseReleaseEvent(QMouseEvent*) override;
    void mouseDoubleClickEvent(QMouseEvent*) override;
private:
    bool projectPt(float wx, float wy, float wz, QPointF& out) const;
    void drawRing3D(QPainter& p, float r, float nx, float ny, float nz, QColor c, int seg = 64);
    void drawCircleAtZ(QPainter& p, float r, float z, QColor c, int seg = 64);

    std::vector<GeodesicRay> m_rays;
    float  m_r_h{2.f}, m_r_ph_lo{3.f}, m_r_ph_hi{3.f};
    float  m_r_isco{6.f}, m_M{1.f};
    float  m_cam_r{20.f}, m_cam_theta{1.5707f}, m_cam_phi{0.f};

    float  m_azi{0.6f}, m_elev{0.4f}, m_dist{50.f};
    QPoint m_lastMouse;
    bool   m_dragging{false};
};

class GeodesicDebugWidget : public QWidget {
    Q_OBJECT
public:
    explicit GeodesicDebugWidget(QWidget* parent = nullptr);
    void setParams(const RenderParams* p);

public slots:
    void trace();

private:
    void appendLog(const QString& s);

    const RenderParams* m_params = nullptr;
    GeodesicView*       m_view;
    Geodesic3DView*     m_view3d;
    QTextEdit*          m_log;
    QSpinBox*           m_spinRays;
    QSpinBox*           m_spinDebugSteps;
    QCheckBox*          m_chkFocus;
    QDoubleSpinBox*     m_spinFocusMin;
    QDoubleSpinBox*     m_spinFocusMax;
    QSpinBox*           m_spinFocusRays;
    QSpinBox*           m_spinMaxOrbits;

    // Sweep mode radio buttons
    QRadioButton*       m_radioHoriz;
    QRadioButton*       m_radioVert;
    QRadioButton*       m_radioLine;
    QRadioButton*       m_radioGrid;

    // 1D sweep panel (horiz / vert)
    QWidget*            m_sweep1DPanel;
    QDoubleSpinBox*     m_spinScanMin;
    QDoubleSpinBox*     m_spinScanMax;
    QDoubleSpinBox*     m_spinFixedNdc;

    // Arbitrary-line sweep panel
    QWidget*            m_sweepLinePanel;
    QDoubleSpinBox*     m_spinLineX1, *m_spinLineY1;
    QDoubleSpinBox*     m_spinLineX2, *m_spinLineY2;

    // 2-D grid sweep panel
    QWidget*            m_sweepGridPanel;
    QDoubleSpinBox*     m_spinGridXMin, *m_spinGridXMax;
    QDoubleSpinBox*     m_spinGridYMin, *m_spinGridYMax;
    QSpinBox*           m_spinGridNx,   *m_spinGridNy;
};
