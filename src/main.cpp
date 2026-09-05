#include <QApplication>
#include <QSurfaceFormat>
#include <QMessageBox>
#include <cuda_runtime.h>
#include "MainWindow.h"

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    app.setApplicationName("BlackHoleSim");
    app.setOrganizationName("Sim");

    // OpenGL 3.3 core profile
    QSurfaceFormat fmt;
    fmt.setVersion(3, 3);
    fmt.setProfile(QSurfaceFormat::CoreProfile);
    fmt.setDepthBufferSize(24);
    fmt.setSamples(1);
    QSurfaceFormat::setDefaultFormat(fmt);

    // Check CUDA
    int devCount = 0;
    if (cudaGetDeviceCount(&devCount) != cudaSuccess || devCount == 0) {
        QMessageBox::critical(nullptr, "CUDA Error",
            "CUDA対応GPUが見つかりませんでした。\n"
            "NVIDIA GPU とCUDAドライバが必要です。");
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    qDebug("CUDA device: %s  (CC %d.%d)", prop.name, prop.major, prop.minor);

    MainWindow win;
    win.show();
    return app.exec();
}
