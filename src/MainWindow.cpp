#include "MainWindow.h"
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QFormLayout>
#include <QGroupBox>
#include <QFileDialog>
#include <QStatusBar>
#include <QMenuBar>
#include <QAction>
#include <QSplitter>
#include <QLabel>
#include <QDateTime>
#include <QSignalBlocker>
#include <cmath>

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setWindowTitle(QString("Black Hole Simulation  —  Kerr-Newman  [build %1]").arg(__DATE__));
    resize(1400, 800);

    // Central: splitter [viewport | controls]
    auto* splitter = new QSplitter(Qt::Horizontal, this);

    m_bh = new BlackHoleWidget;
    m_bh->setMinimumSize(640, 480);
    splitter->addWidget(m_bh);
    splitter->addWidget(buildControlPanel());
    splitter->setStretchFactor(0, 3);
    splitter->setStretchFactor(1, 1);
    setCentralWidget(splitter);

    // Status bar
    m_lblFPS          = new QLabel("FPS: --");
    m_lblTimeDilation = new QLabel("時間遅れ: --");
    m_lblMode         = new QLabel(m_bh->isUsingCuda() ? "GPU" : "CPU (低速モード)");
    statusBar()->addPermanentWidget(m_lblFPS);
    statusBar()->addPermanentWidget(m_lblTimeDilation);
    statusBar()->addPermanentWidget(m_lblMode);

    // Geodesic debug dock (bottom, hidden by default)
    m_debugWidget = new GeodesicDebugWidget;
    m_debugWidget->setParams(&m_bh->params());
    m_debugDock = new QDockWidget("測地線デバッグ  (CPU積分・保存量チェック)", this);
    m_debugDock->setWidget(m_debugWidget);
    m_debugDock->setMinimumHeight(350);
    addDockWidget(Qt::BottomDockWidgetArea, m_debugDock);
    m_debugDock->setVisible(false);

    // Menu
    auto* file = menuBar()->addMenu("ファイル");
    auto* actSave = file->addAction("画像を保存 (PNG)");
    auto* actLoad = file->addAction("背景テクスチャを読み込む");
    connect(actSave, &QAction::triggered, this, &MainWindow::onSaveImage);
    connect(actLoad, &QAction::triggered, this, &MainWindow::onLoadTexture);

    auto* dbgMenu  = menuBar()->addMenu("デバッグ");
    auto* actDebug = dbgMenu->addAction("測地線デバッグパネル");
    actDebug->setCheckable(true);
    actDebug->setShortcut(QKeySequence("Ctrl+D"));
    connect(actDebug, &QAction::toggled, m_debugDock, &QDockWidget::setVisible);
    connect(m_debugDock, &QDockWidget::visibilityChanged, actDebug, &QAction::setChecked);

    // Sync params pointer whenever BH widget renders (params may change)
    connect(m_bh, &BlackHoleWidget::fpsUpdated, this, [this](){
        m_debugWidget->setParams(&m_bh->params());
    });

    connect(m_bh, &BlackHoleWidget::fpsUpdated,          this, &MainWindow::updateFPS);
    connect(m_bh, &BlackHoleWidget::timeDilationUpdated, this, &MainWindow::updateTimeDilation);
}

QWidget* MainWindow::buildControlPanel() {
    auto* panel = new QWidget;
    auto* vlay  = new QVBoxLayout(panel);
    vlay->setSpacing(8);
    panel->setFixedWidth(280);

    vlay->addWidget(buildBHGroup());
    vlay->addWidget(buildCameraGroup());
    vlay->addWidget(buildRenderGroup());
    vlay->addStretch();
    return panel;
}

QGroupBox* MainWindow::buildBHGroup() {
    auto* grp  = new QGroupBox("ブラックホール パラメータ");
    auto* form = new QFormLayout(grp);

    m_spinMass = new QDoubleSpinBox;
    m_spinMass->setRange(0.1, 100.0);
    m_spinMass->setValue(1.0);
    m_spinMass->setSingleStep(0.1);
    m_spinMass->setToolTip("質量 M（幾何単位）");
    form->addRow("質量 M:", m_spinMass);

    m_spinSpin = new QDoubleSpinBox;
    m_spinSpin->setRange(-0.999, 0.999);
    m_spinSpin->setValue(0.0);
    m_spinSpin->setSingleStep(0.05);
    m_spinSpin->setToolTip("角運動量パラメータ a/M  (-1 ～ 1)");
    form->addRow("スピン a/M:", m_spinSpin);

    m_spinCharge = new QDoubleSpinBox;
    m_spinCharge->setRange(-0.999, 0.999);
    m_spinCharge->setValue(0.0);
    m_spinCharge->setSingleStep(0.05);
    m_spinCharge->setToolTip("電荷パラメータ Q/M  (-1 ～ 1)");
    form->addRow("電荷 Q/M:", m_spinCharge);

    m_chkDisk = new QCheckBox("降着円盤を表示");
    m_chkDisk->setChecked(true);
    form->addRow(m_chkDisk);

    // Enforce a² + Q² ≤ M² (宇宙検閲条件 — 裸の特異点を禁止)
    // When spin changes, clamp and update charge range, then sync BH
    connect(m_spinSpin, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, [this](double aOverM) {
        double maxQ = std::sqrt(std::max(0.0, 1.0 - aOverM * aOverM)) * 0.9999;
        {
            QSignalBlocker blk(m_spinCharge);
            m_spinCharge->setRange(-maxQ, maxQ);
        }
        m_bh->setCharge(m_spinCharge->value());
    });
    // When charge changes, clamp and update spin range, then sync BH
    connect(m_spinCharge, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, [this](double qOverM) {
        double maxA = std::sqrt(std::max(0.0, 1.0 - qOverM * qOverM)) * 0.9999;
        {
            QSignalBlocker blk(m_spinSpin);
            m_spinSpin->setRange(-maxA, maxA);
        }
        m_bh->setSpin(m_spinSpin->value());
    });

    connect(m_spinMass,   QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            m_bh, &BlackHoleWidget::setMass);
    connect(m_spinSpin,   QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            m_bh, &BlackHoleWidget::setSpin);
    connect(m_spinCharge, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            m_bh, &BlackHoleWidget::setCharge);
    connect(m_chkDisk,    &QCheckBox::toggled,
            m_bh, &BlackHoleWidget::setShowDisk);

    return grp;
}

QGroupBox* MainWindow::buildCameraGroup() {
    auto* grp  = new QGroupBox("カメラ");
    auto* vlay = new QVBoxLayout(grp);

    auto* btnReset = new QPushButton("カメラリセット");
    connect(btnReset, &QPushButton::clicked, m_bh, &BlackHoleWidget::resetCamera);
    vlay->addWidget(btnReset);

    m_chkAnimate = new QCheckBox("カメラ自動周回アニメーション");
    m_chkAnimate->setChecked(false);
    connect(m_chkAnimate, &QCheckBox::toggled, m_bh, &BlackHoleWidget::setAnimating);
    vlay->addWidget(m_chkAnimate);

    vlay->addWidget(new QLabel("マウス左ドラッグ: 視点回転\n"
                               "マウス右ドラッグ: カメラロール\n"
                               "ホイール: ズーム"));
    return grp;
}

QGroupBox* MainWindow::buildRenderGroup() {
    auto* grp  = new QGroupBox("レンダリング設定");
    auto* form = new QFormLayout(grp);

    m_spinSteps = new QSpinBox;
    m_spinSteps->setRange(50, 5000);
    m_spinSteps->setValue(600);
    m_spinSteps->setSingleStep(50);
    m_spinSteps->setToolTip(
        "測地線積分の最大ステップ数。\n"
        "計算品質スライダーで自動調整されます（手動上書き可）。");
    form->addRow("積分ステップ:", m_spinSteps);
    connect(m_spinSteps, QOverload<int>::of(&QSpinBox::valueChanged),
            m_bh, &BlackHoleWidget::setMaxSteps);

    // Quality/speed slider: 1=speed  3=balance  5=precision
    // step_size and max_steps are set together so total Mino-time coverage stays
    // constant regardless of quality: N ∝ ln(r_cam/r_h) / step_size.
    // Reference (quality=3): step=0.05, 600 steps.  Scale: N = 600*(0.05/step).
    auto* qualRow = new QWidget;
    auto* ql = new QHBoxLayout(qualRow);
    ql->setContentsMargins(0, 0, 0, 0);
    ql->addWidget(new QLabel("速度"));
    m_sliderQuality = new QSlider(Qt::Horizontal);
    m_sliderQuality->setRange(1, 5);
    m_sliderQuality->setValue(3);
    m_sliderQuality->setTickPosition(QSlider::TicksBelow);
    m_sliderQuality->setTickInterval(1);
    m_sliderQuality->setToolTip(
        "1=高速 (step=0.20, 150steps)\n"
        "2      (step=0.10, 300steps)\n"
        "3=標準 (step=0.05, 600steps)\n"
        "4      (step=0.025, 1200steps)\n"
        "5=高精度(step=0.01, 3000steps)\n"
        "積分ステップ数はカバレッジ一定になるよう自動調整されます。");
    ql->addWidget(m_sliderQuality);
    ql->addWidget(new QLabel("精度"));
    form->addRow("計算品質:", qualRow);

    connect(m_sliderQuality, &QSlider::valueChanged, this, [this](int v) {
        static const double stepSizes[] = {0.20, 0.10, 0.05, 0.025, 0.01};
        // Scale max_steps so total Mino-time coverage is constant across all quality levels.
        // Without this, quality=5 (step=0.01) exhausts 600 steps before reaching the
        // horizon from r=20, incorrectly classifying absorbed rays as escaped.
        static const int    maxSteps[]  = { 150,  300,  600,  1200,  3000};
        m_bh->setStepSize(stepSizes[v - 1]);
        m_bh->setMaxSteps(maxSteps[v - 1]);
        QSignalBlocker blk(m_spinSteps);
        m_spinSteps->setValue(maxSteps[v - 1]);
    });

    m_comboDebug = new QComboBox;
    m_comboDebug->addItem("0: 通常",              0);
    m_comboDebug->addItem("1: 終了状態",          1);
    m_comboDebug->addItem("3: Seam 可視化",       3);
    m_comboDebug->addItem("4: UV 偽カラー (cos/sin)",  4);
    m_comboDebug->addItem("5: 経緯度グリッド",         5);
    m_comboDebug->setToolTip(
        "0: 通常レンダリング\n"
        "1: 終了状態 (紫=NaN 赤=ステップ切れ 黒=地平線 黄=UV不正 緑=極点 青=r折返 水色=θ折返)\n"
        "3: Seam 可視化 (マゼンタ=u≈0/1, 黒=それ以外)\n"
        "4: UV 偽カラー — 赤=cos(2πu), 緑=sin(2πu), 青=v\n"
        "   S^1表現なので経度seam でも連続; 不連続なら別のUV異常\n"
        "5: 経緯度グリッド (手続き型) — circular distance で u=0≡u=1\n"
        "   ブラックホール方向の縦線が一本に重なることを確認");
    form->addRow("デバッグ表示:", m_comboDebug);
    connect(m_comboDebug, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, [this](int idx) {
        m_bh->setDebugMode(m_comboDebug->itemData(idx).toInt());
    });

    auto* btnSave = new QPushButton("PNG 画像を保存");
    connect(btnSave, &QPushButton::clicked, this, &MainWindow::onSaveImage);
    form->addRow(btnSave);

    auto* btnTex = new QPushButton("背景テクスチャ変更");
    connect(btnTex, &QPushButton::clicked, this, &MainWindow::onLoadTexture);
    form->addRow(btnTex);

    return grp;
}

void MainWindow::onSaveImage() {
    QString path = QFileDialog::getSaveFileName(this,
        "画像を保存", "blackhole_" + QDateTime::currentDateTime().toString("yyyyMMdd_hhmmss") + ".png",
        "PNG (*.png)");
    if (!path.isEmpty()) m_bh->saveFrameToFile(path);
}

void MainWindow::onLoadTexture() {
    QString path = QFileDialog::getOpenFileName(this,
        "テクスチャを選択", "",
        "画像ファイル (*.jpg *.jpeg *.png *.exr)");
    if (!path.isEmpty()) m_bh->setBackgroundTexture(path);
}

void MainWindow::updateFPS(double fps) {
    m_lblFPS->setText(QString("FPS: %1").arg(fps, 0, 'f', 1));
}

void MainWindow::updateTimeDilation(double factor) {
    // Switch to scientific notation when factor drops below the precision
    // of the fixed-point display (5 decimal places → 3 sig figs at 0.001).
    QString fStr, pStr;
    if (factor >= 1e-3) {
        fStr = QString::number(factor,        'f', 5);
        pStr = QString::number(factor * 100.0,'f', 2) + "%";
    } else {
        fStr = QString::number(factor,        'e', 3);
        pStr = QString::number(factor * 100.0,'e', 2) + "%";
    }
    m_lblTimeDilation->setText(
        QString("時間遅れ係数: %1  (遠方比 %2)").arg(fStr).arg(pStr));
}
