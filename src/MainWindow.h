#pragma once
#include <QMainWindow>
#include <QLabel>
#include <QDoubleSpinBox>
#include <QSlider>
#include <QCheckBox>
#include <QPushButton>
#include <QSpinBox>
#include <QComboBox>
#include <QGroupBox>
#include <QDockWidget>
#include <QSlider>
#include "BlackHoleWidget.h"
#include "GeodesicDebugWidget.h"

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);

private slots:
    void onSaveImage();
    void onLoadTexture();
    void updateFPS(double fps);
    void updateTimeDilation(double factor);

private:
    QWidget*        buildControlPanel();
    QGroupBox*      buildBHGroup();
    QGroupBox*      buildCameraGroup();
    QGroupBox*      buildRenderGroup();

    BlackHoleWidget*      m_bh;
    GeodesicDebugWidget*  m_debugWidget;
    QDockWidget*          m_debugDock;
    QLabel*          m_lblFPS;
    QLabel*          m_lblTimeDilation;
    QLabel*          m_lblMode;
    QDoubleSpinBox*  m_spinMass;
    QDoubleSpinBox*  m_spinSpin;
    QDoubleSpinBox*  m_spinCharge;
    QCheckBox*       m_chkDisk;
    QCheckBox*       m_chkAnimate;
    QSpinBox*        m_spinSteps;
    QSlider*         m_sliderQuality;
    QComboBox*       m_comboDebug;
};
