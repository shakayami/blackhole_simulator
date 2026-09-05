# Blackhole Simulator

一般相対性理論（Kerr-Newman 計量）に基づき、光子の測地線を GPU（CUDA）でリアルタイムに追跡してブラックホールの見え方を描画する C++ / Qt / CUDA 製シミュレータです。

質量・スピン（角運動量）・電荷を自由に変更しながら、重力レンズ効果によって背景の天球テクスチャや降着円盤がどう歪んで見えるかを対話的に観察できます。

## 特徴

- **Kerr-Newman ブラックホール**：質量 M・スピン a/M・電荷 Q/M をリアルタイムに変更可能
- **一般相対論的レイトレーシング**：光子の測地線方程式を数値積分し、重力レンズによる背景の歪みを再現
- **降着円盤の表示 / 非表示切り替え**
- **背景天球テクスチャ**：NASA 提供の高解像度スターマップを使用（未配置時はプロシージャル生成の星空にフォールバック）
- **GPU (CUDA) レンダリング**：CUDA 対応 GPU がない場合は CPU レンダラーにフォールバック
- **TAA（時間的アンチエイリアシング）によるノイズ低減**、積分ステップ数による精度調整
- **カメラ操作**：軌道回転・ロール・ズームによる自由な視点変更
- **時間の遅れ（重力による時間の遅れ）の表示**
- **測地線デバッグ表示**、現在フレームの PNG 画像保存
- Qt Widgets によるパラメータ調整 GUI

## 動作環境

| ツール | バージョン |
|--------|-----------|
| Windows | 10 / 11 (64-bit) |
| Visual Studio 2022 | 17.x（C++ によるデスクトップ開発ワークロード） |
| CMake | 3.20 以上 |
| CUDA Toolkit | 12.x |
| Qt | 6.6 以上（MSVC 2022 64-bit） |
| vcpkg | 最新 |

> **GPU 要件**: NVIDIA GPU（Compute Capability 7.5 以上、RTX 20xx / A100 / RTX 30xx / RTX 40xx 世代以降）が必要です。CUDA 対応 GPU が見つからない場合、起動時にエラーダイアログが表示されます。

詳しいビルド手順・環境構築（vcpkg のセットアップ、パス設定、CUDA アーキテクチャ調整など）は [BUILD.md](BUILD.md) を参照してください。

## クイックスタート

```powershell
# 1. 依存関係を導入した vcpkg / Qt / CUDA Toolkit をセットアップ済みであること
# 2. CMakePresets.json のパス（vcpkg・Qt）を環境に合わせて編集
cmake --preset windows-release
cmake --build --preset windows-release

# 実行
.\build\Release\BlackholeSim.exe
```

## 操作方法

| 操作 | 動作 |
|------|------|
| 左ドラッグ | 視点回転（軌道） |
| 右ドラッグ | カメラロール |
| ホイール | ズーム |
| 質量 M | ブラックホールの質量（シュワルツシルト半径 rs = 2M） |
| スピン a/M | Kerr 回転パラメータ（−1〜1） |
| 電荷 Q/M | Kerr-Newman 電荷（−1〜1） |
| 降着円盤 | 表示 / 非表示トグル |
| 積分ステップ | 精度調整（多いほど重い） |
| PNG 保存 | 現在フレームを画像出力 |

## 背景テクスチャについて

背景の天球には NASA Scientific Visualization Studio の "Deep Star Maps"（`starmap_2020_*`）などのテクスチャを使用します。サイズが大きく、また再配布ライセンスが未確認のためリポジトリには含めていません。セットアップ方法は [assets/textures/README.md](assets/textures/README.md) を参照してください。テクスチャを配置しない場合はプロシージャル生成の星空にフォールバックします。

## ディレクトリ構成

```
src/
  main.cpp              # エントリポイント（CUDA デバイスチェック含む）
  MainWindow.*           # メインウィンドウ・パラメータ操作 GUI
  BlackHoleWidget.*       # OpenGL 描画ウィジェット（CUDA/CPU レンダラーの呼び出し）
  CpuRenderer.*           # CUDA 非搭載環境向け CPU レンダラー
  TextureLoader.*         # テクスチャ読み込み（stb 使用）
  GeodesicDebugWidget.*   # 測地線デバッグ表示
  Camera.h                # カメラ制御
  physics/KerrNewman.h    # Kerr-Newman 計量・測地線方程式
  cuda/RayTracer.cu(h)    # CUDA によるレイトレーシング本体
assets/textures/          # 背景天球テクスチャ（git 管理外、README 参照）
resources/                # Qt リソースファイル
```

## 使用技術

- C++17 / CUDA
- Qt 6（Core, Widgets, OpenGL, OpenGLWidgets）
- CMake / CMakePresets / vcpkg
- glm, stb（vcpkg 経由）

## ライセンス

本リポジトリのソースコードは [MIT License](LICENSE) の下で公開しています。

Qt は LGPL/商用ライセンスで別途配布されており、本リポジトリには含まれていません（vcpkg / 各自の Qt インストールを利用）。バイナリを配布する場合は Qt 側のライセンス条件（LGPL の動的リンク要件など）を別途確認してください。背景テクスチャ（NASA Deep Star Maps）もリポジトリには含めていません（[assets/textures/README.md](assets/textures/README.md) 参照）。
