#pragma once
#include <QOpenGLWidget>
#include <QOpenGLFunctions_3_3_Core>
#include <QOpenGLShaderProgram>
#include <QTimer>
#include <QElapsedTimer>
#include <cuda_runtime.h>
#include <vector>
#include "Camera.h"
#include "TextureLoader.h"
#include "cuda/RayTracer.cuh"

class BlackHoleWidget : public QOpenGLWidget, protected QOpenGLFunctions_3_3_Core {
    Q_OBJECT
public:
    explicit BlackHoleWidget(QWidget* parent = nullptr);
    ~BlackHoleWidget() override;

    Camera&      camera()    { return m_cam; }
    RenderParams& params()   { return m_params; }

    bool isUsingCuda() const { return m_useCuda; }

    void setBackgroundTexture(const QString& path);
    void saveFrameToFile(const QString& path);

    double timeDilationFactor() const;

public slots:
    void setMass(double v);
    void setSpin(double v);
    void setCharge(double v);
    void setShowDisk(bool v);
    void setAnimating(bool v);
    void setMaxSteps(int v);
    void setStepSize(double v);
    void setDebugMode(int v);
    void resetCamera();

signals:
    void fpsUpdated(double fps);
    void timeDilationUpdated(double factor);

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;
    void mousePressEvent(QMouseEvent*) override;
    void mouseMoveEvent(QMouseEvent*) override;
    void wheelEvent(QWheelEvent*) override;

private:
    void initCUDA();
    void releaseCUDA();
    void renderFrame();
    void uploadToGL(const float4* hostPixels);

    bool m_useCuda = true;

    // TAA state
    int   m_accumFrames  = 0;
    float m_prevCamR     = -1.0f, m_prevCamTheta = -1.0f, m_prevCamPhi = -1.0f;
    float m_prevLookYaw  = 0.0f,  m_prevLookPitch = 0.0f;
    float m_prevBhM      = -1.0f, m_prevBhA = 0.0f, m_prevBhQ = 0.0f;
    bool  m_prevShowDisk = true;

    // GL
    GLuint              m_tex  = 0;
    GLuint              m_vao  = 0;
    GLuint              m_vbo  = 0;
    QOpenGLShaderProgram m_prog;

    // GPU buffers (used when m_useCuda)
    float4*             d_output = nullptr;
    float4*             d_accum  = nullptr;

    // CPU buffers (used when !m_useCuda)
    std::vector<float4> m_cpuOutput;
    std::vector<float4> m_cpuAccum;

    int                 m_bufW = 0, m_bufH = 0;

    // Textures
    CudaTexture         m_bgTex;
    CudaTexture         m_diskTex;

    // State
    Camera              m_cam;
    RenderParams        m_params;
    QPoint              m_lastMouse;
    bool                m_animating = false;
    float               m_animPhase = 0.0f;

    QTimer              m_timer;
    QElapsedTimer       m_elapsed;
    int                 m_frameCount = 0;
};
