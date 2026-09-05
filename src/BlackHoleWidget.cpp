#include "BlackHoleWidget.h"
#include "TextureLoader.h"
#include "CpuRenderer.h"
#include "physics/KerrNewman.h"
#include <QMouseEvent>
#include <QWheelEvent>
#include <QCoreApplication>
#include <QDir>
#include <QOpenGLFunctions_3_3_Core>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

// Full-screen quad vertices (pos, uv)
static const float kQuadVerts[] = {
    -1,-1, 0,1,
     1,-1, 1,1,
     1, 1, 1,0,
    -1,-1, 0,1,
     1, 1, 1,0,
    -1, 1, 0,0,
};

BlackHoleWidget::BlackHoleWidget(QWidget* parent) : QOpenGLWidget(parent) {
    // Detect CUDA device availability before any texture work.
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    m_useCuda = (deviceCount > 0);
    setUseCuda(m_useCuda);   // tell TextureLoader which path to use

    m_params.bh.M       = 1.0f;
    m_params.bh.a       = 0.0f;
    m_params.bh.Q       = 0.0f;
    m_params.show_disk  = true;
    m_params.disk_inner = KN::r_isco(m_params.bh.M, m_params.bh.a);
    m_params.disk_outer = 20.0f;
    m_params.max_steps      = 600;
    m_params.step_size      = 0.05f;
    m_params.fov            = 1.0f;
    m_params.sky_lon_offset = 0.1234567f; // Deliberately non-round: keep grid meridians and atan2 seam off the BH/camera axis
    m_params.debug_mode     = 0;

    connect(&m_timer, &QTimer::timeout, this, [this](){
        if (m_animating) {
            m_animPhase += 0.005f;
            m_cam.phi = m_animPhase;
        }
        update();
    });
    m_timer.start(16);
    m_elapsed.start();
}

BlackHoleWidget::~BlackHoleWidget() {
    makeCurrent();
    releaseCUDA();
    if (m_tex) glDeleteTextures(1, &m_tex);
    if (m_vao) glDeleteVertexArrays(1, &m_vao);
    if (m_vbo) glDeleteBuffers(1, &m_vbo);
    doneCurrent();
}

void BlackHoleWidget::initializeGL() {
    initializeOpenGLFunctions();
    glClearColor(0, 0, 0, 1);

    // Shader: blit texture with gamma correction
    m_prog.addShaderFromSourceCode(QOpenGLShader::Vertex, R"(
        #version 330 core
        layout(location=0) in vec2 pos;
        layout(location=1) in vec2 uv;
        out vec2 vUV;
        void main() { gl_Position = vec4(pos, 0, 1); vUV = uv; }
    )");
    m_prog.addShaderFromSourceCode(QOpenGLShader::Fragment, R"(
        #version 330 core
        in vec2 vUV;
        out vec4 fragColor;
        uniform sampler2D uTex;
        uniform float uGamma;
        void main() {
            vec4 c = texture(uTex, vUV);
            c.rgb = pow(clamp(c.rgb, 0.0, 1.0), vec3(1.0 / uGamma));
            fragColor = c;
        }
    )");
    m_prog.link();

    // VAO/VBO
    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);
    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(kQuadVerts), kQuadVerts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)(2*sizeof(float)));
    glBindVertexArray(0);

    // GL texture (upscaled bilinear when CPU render is at half resolution)
    glGenTextures(1, &m_tex);
    glBindTexture(GL_TEXTURE_2D, m_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Load background starmap; fall back to procedural starfield if none found.
    {
        QStringList dirs;
        dirs << QCoreApplication::applicationDirPath()
             << QDir(QCoreApplication::applicationDirPath()).absoluteFilePath("..")
             << QDir::currentPath();
        dirs.removeDuplicates();

        QStringList patterns = {
            "starmap*8k*.png", "starmap*8k*.jpg",
            "starmap*4k*.png", "starmap*4k*.jpg",
            "starmap*.png",    "starmap*.jpg"
        };

        bool loaded_ok = false;
        for (const QString& dir : dirs) {
            if (loaded_ok) break;
            QDir qdir(dir);
            QStringList candidates;
            QSet<QString> seen;
            for (const QString& pat : patterns) {
                for (const QString& name : qdir.entryList({pat}, QDir::Files)) {
                    QString abs = qdir.absoluteFilePath(name);
                    if (!seen.contains(abs)) { seen.insert(abs); candidates << abs; }
                }
            }
            for (const QString& path : candidates) {
                CudaTexture loaded = loadTexture(path.toStdString());
                if (loaded.valid() && loaded.width >= 64) {
                    m_bgTex   = loaded;
                    loaded_ok = true;
                    break;
                }
                loaded.free();
            }
        }
        if (!loaded_ok)
            m_bgTex = makeStarfieldTexture(2048, 1024);
    }

    m_diskTex = makeSolidTexture(1.0f, 0.8f, 0.4f, 1.0f);

    initCUDA();
}

void BlackHoleWidget::resizeGL(int w, int h) {
    glViewport(0, 0, w, h);
    releaseCUDA();
    initCUDA();
}

// Low-discrepancy Halton sequence for TAA jitter
static float halton(int idx, int base) {
    float f = 1.0f, r = 0.0f;
    while (idx > 0) { f /= base; r += f * (idx % base); idx /= base; }
    return r;
}

void BlackHoleWidget::initCUDA() {
    int w = width() > 0 ? width() : 800;
    int h = height() > 0 ? height() : 600;

    if (!m_useCuda) {
        // Render at half linear resolution to keep CPU frame time manageable.
        // GL_LINEAR upscaling fills the window without visible nearest-neighbour aliasing.
        w = std::max(w / 2, 64);
        h = std::max(h / 2, 64);
    }

    m_bufW = w; m_bufH = h;

    if (m_useCuda) {
        cudaMalloc(&d_output, w * h * sizeof(float4));
        cudaMalloc(&d_accum,  w * h * sizeof(float4));
        cudaMemset(d_accum, 0, w * h * sizeof(float4));
    } else {
        m_cpuOutput.assign(w * h, {0,0,0,0});
        m_cpuAccum.assign(w * h, {0,0,0,0});
    }
    m_accumFrames = 0;
}

void BlackHoleWidget::releaseCUDA() {
    if (d_output) { cudaFree(d_output); d_output = nullptr; }
    if (d_accum)  { cudaFree(d_accum);  d_accum  = nullptr; }
    m_cpuOutput.clear();
    m_cpuAccum.clear();
}

void BlackHoleWidget::setBackgroundTexture(const QString& path) {
    if (m_bgTex.valid()) m_bgTex.free();
    m_bgTex = loadTexture(path.toStdString());
}

void BlackHoleWidget::paintGL() {
    if (!m_bgTex.valid()) return;
    if ( m_useCuda && !d_output)           return;
    if (!m_useCuda && m_cpuOutput.empty()) return;

    // Keep camera horizon in sync with current BH params
    float r_h = KN::r_plus(m_params.bh.M, m_params.bh.a, m_params.bh.Q);
    if (r_h != m_cam.r_horizon) {
        m_cam.r_horizon = r_h;
        m_cam.clampToHorizon();
    }

    m_params.width     = m_bufW;
    m_params.height    = m_bufH;
    m_params.cam_r          = m_cam.r;
    m_params.cam_theta      = m_cam.theta;
    m_params.cam_phi        = m_cam.phi;
    m_params.cam_up_phi     = m_cam.up_phi;
    m_params.cam_look_yaw   = m_cam.look_yaw;
    m_params.cam_look_pitch = m_cam.look_pitch;
    m_params.fov            = m_cam.fov;

    // TAA: detect scene change to reset accumulation
    bool changed = (m_params.cam_r          != m_prevCamR     ||
                    m_params.cam_theta       != m_prevCamTheta  ||
                    m_params.cam_phi         != m_prevCamPhi    ||
                    m_params.cam_look_yaw    != m_prevLookYaw   ||
                    m_params.cam_look_pitch  != m_prevLookPitch ||
                    m_params.bh.M            != m_prevBhM       ||
                    m_params.bh.a            != m_prevBhA       ||
                    m_params.bh.Q            != m_prevBhQ       ||
                    m_params.show_disk       != m_prevShowDisk);
    if (changed) {
        m_accumFrames = 0;
        m_prevCamR      = m_params.cam_r;
        m_prevCamTheta  = m_params.cam_theta;
        m_prevCamPhi    = m_params.cam_phi;
        m_prevLookYaw   = m_params.cam_look_yaw;
        m_prevLookPitch = m_params.cam_look_pitch;
        m_prevBhM       = m_params.bh.M;
        m_prevBhA       = m_params.bh.a;
        m_prevBhQ       = m_params.bh.Q;
        m_prevShowDisk  = m_params.show_disk;
    }

    if (m_accumFrames > 0) {
        int idx = m_accumFrames % 128;
        m_params.jitter_x = (halton(idx, 2) - 0.5f) * (2.0f / m_bufW);
        m_params.jitter_y = (halton(idx, 3) - 0.5f) * (2.0f / m_bufH);
    } else {
        m_params.jitter_x = m_params.jitter_y = 0.0f;
    }

    float alpha = 1.0f / std::min(float(m_accumFrames + 1), 32.0f);

    if (m_useCuda) {
        launchRayTracer(d_output, m_bgTex.tex, m_diskTex.tex, m_params);
        cudaDeviceSynchronize();
        launchTemporalBlend(d_accum, d_output, m_bufW, m_bufH, alpha);
        cudaDeviceSynchronize();

        std::vector<float4> tmp(m_bufW * m_bufH);
        cudaMemcpy(tmp.data(), d_accum,
                   m_bufW * m_bufH * sizeof(float4), cudaMemcpyDeviceToHost);
        uploadToGL(tmp.data());
    } else {
        launchRayTracerCPU(m_cpuOutput.data(), m_bgTex, m_diskTex, m_params);
        launchTemporalBlendCPU(m_cpuAccum.data(), m_cpuOutput.data(), m_bufW, m_bufH, alpha);
        uploadToGL(m_cpuAccum.data());
    }
    ++m_accumFrames;

    // Draw fullscreen quad
    glClear(GL_COLOR_BUFFER_BIT);
    m_prog.bind();
    m_prog.setUniformValue("uTex",   0);
    m_prog.setUniformValue("uGamma", 2.2f);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_tex);
    glBindVertexArray(m_vao);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);
    m_prog.release();

    // FPS
    ++m_frameCount;
    qint64 ms = m_elapsed.elapsed();
    if (ms >= 1000) {
        double fps = m_frameCount * 1000.0 / ms;
        emit fpsUpdated(fps);
        emit timeDilationUpdated(timeDilationFactor());
        m_frameCount = 0;
        m_elapsed.restart();
    }
}

void BlackHoleWidget::uploadToGL(const float4* hostPixels) {
    glBindTexture(GL_TEXTURE_2D, m_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, m_bufW, m_bufH, 0,
                 GL_RGBA, GL_FLOAT, hostPixels);
}

double BlackHoleWidget::timeDilationFactor() const {
    double r   = m_cam.r;
    double M   = m_params.bh.M;
    double a   = m_params.bh.a;
    double Q   = m_params.bh.Q;
    double ct  = std::cos(m_params.cam_theta);
    double sig = r*r + a*a*ct*ct;
    double val = 1.0 - (2.0*M*r - Q*Q) / sig;
    if (val <= 0.0) return 0.0;
    return std::sqrt(val);
}

void BlackHoleWidget::saveFrameToFile(const QString& path) {
    std::vector<float4> pixels;

    if (m_useCuda) {
        if (!d_accum) return;
        pixels.resize(m_bufW * m_bufH);
        cudaMemcpy(pixels.data(), d_accum,
                   m_bufW * m_bufH * sizeof(float4), cudaMemcpyDeviceToHost);
    } else {
        if (m_cpuAccum.empty()) return;
        pixels = m_cpuAccum;
    }

    // Apply the same gamma correction as the display shader (pow(x, 1/2.2))
    auto toU8 = [](float v) -> uint8_t {
        v = std::pow(std::max(0.0f, std::min(1.0f, v)), 1.0f / 2.2f);
        return static_cast<uint8_t>(v * 255.0f + 0.5f);
    };

    QImage img(m_bufW, m_bufH, QImage::Format_RGBA8888);
    for (int y = 0; y < m_bufH; ++y) {
        uint8_t* row = img.scanLine(y);
        for (int x = 0; x < m_bufW; ++x) {
            int i = y * m_bufW + x;
            row[x*4+0] = toU8(pixels[i].x);
            row[x*4+1] = toU8(pixels[i].y);
            row[x*4+2] = toU8(pixels[i].z);
            row[x*4+3] = 255;
        }
    }
    img.save(path);
}

// ── Slots ─────────────────────────────────────────────────────────────────────

void BlackHoleWidget::setMass(double v)     { m_params.bh.M = (float)v; }
void BlackHoleWidget::setSpin(double v) {
    double q_norm = (m_params.bh.M > 0.0f) ? (m_params.bh.Q / m_params.bh.M) : 0.0;
    double max_a  = std::sqrt(std::max(0.0, 1.0 - q_norm * q_norm)) * (1.0 - 1e-5);
    m_params.bh.a = (float)(std::max(-max_a, std::min(max_a, v)) * m_params.bh.M);
    m_params.disk_inner = KN::r_isco(m_params.bh.M, m_params.bh.a);
}
void BlackHoleWidget::setCharge(double v) {
    double a_norm = (m_params.bh.M > 0.0f) ? (m_params.bh.a / m_params.bh.M) : 0.0;
    double max_q  = std::sqrt(std::max(0.0, 1.0 - a_norm * a_norm)) * (1.0 - 1e-5);
    m_params.bh.Q = (float)(std::max(-max_q, std::min(max_q, v)) * m_params.bh.M);
}
void BlackHoleWidget::setShowDisk(bool v)   { m_params.show_disk = v; }
void BlackHoleWidget::setMaxSteps(int v)    { m_params.max_steps = v; }
void BlackHoleWidget::setStepSize(double v) { m_params.step_size = (float)v; m_accumFrames = 0; }
void BlackHoleWidget::setDebugMode(int v)   { m_params.debug_mode = v; m_accumFrames = 0; }
void BlackHoleWidget::setAnimating(bool v)  { m_animating = v; }
void BlackHoleWidget::resetCamera()         { m_cam = Camera{}; }

// ── Input ─────────────────────────────────────────────────────────────────────

void BlackHoleWidget::mousePressEvent(QMouseEvent* e) {
    m_lastMouse = e->pos();
}

void BlackHoleWidget::mouseMoveEvent(QMouseEvent* e) {
    QPoint delta = e->pos() - m_lastMouse;
    m_lastMouse = e->pos();
    if (e->buttons() & Qt::LeftButton) {
        m_cam.pan(-delta.x() * 0.005f, delta.y() * 0.005f);
    }
    if (e->buttons() & Qt::MiddleButton || e->buttons() & Qt::RightButton) {
        m_cam.orbit(-delta.y() * 0.005f, delta.x() * 0.005f);
    }
}

void BlackHoleWidget::wheelEvent(QWheelEvent* e) {
    float factor = (e->angleDelta().y() > 0) ? 0.9f : 1.1f;
    m_cam.zoom(factor);
}
