# ビルド手順

## 必要なソフトウェア

| ツール | バージョン | 入手先 |
|--------|-----------|--------|
| Windows | 10 / 11 (64-bit) | — |
| Visual Studio 2022 | 17.x（**C++ によるデスクトップ開発**ワークロード必須） | [visualstudio.microsoft.com](https://visualstudio.microsoft.com/) |
| CMake | 3.20 以上 | [cmake.org](https://cmake.org/download/) ／ VS インストーラ同梱版でも可 |
| CUDA Toolkit | 12.x（12.0〜12.x） | [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) |
| Qt | 6.6 以上（MSVC 2022 64-bit ビルド） | [Qt Online Installer](https://www.qt.io/download-qt-installer) |
| vcpkg | 最新 | [github.com/microsoft/vcpkg](https://github.com/microsoft/vcpkg) |
| Git | 任意 | [git-scm.com](https://git-scm.com/) |

> **GPU 要件**: NVIDIA GPU（Compute Capability 7.5 以上）が必要です。
> 対応アーキテクチャ: RTX 20xx (sm_75), A100 (sm_80), RTX 30xx (sm_86), RTX 40xx (sm_89)

---

## 1. vcpkg のセットアップ（初回のみ）

任意のフォルダへクローンしてください（例: `C:\vcpkg`）。

```powershell
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
```

---

## 2. CMakePresets.json の環境依存パスを編集する

`CMakePresets.json` には **自分の環境に合わせて書き換えが必要なパスが 2 か所**あります。

```json
"CMAKE_TOOLCHAIN_FILE": "<vcpkg のクローン先>/scripts/buildsystems/vcpkg.cmake",
"CMAKE_PREFIX_PATH":    "<Qt のインストール先>/<バージョン>/msvc2022_64"
```

### 書き換え例

| 項目 | 例 |
|------|----|
| vcpkg を `C:\vcpkg` へクローンした場合 | `C:/vcpkg/scripts/buildsystems/vcpkg.cmake` |
| Qt 6.8.0 を `C:\Qt` へインストールした場合 | `C:/Qt/6.8.0/msvc2022_64` |

> パス区切りは `\` でなく `/` を使ってください（CMake の仕様）。

`windows-debug` プリセットも同様に修正してください。

---

## 3. CUDA アーキテクチャの確認（通常は変更不要）

`CMakeLists.txt` の `CMAKE_CUDA_ARCHITECTURES` には主要アーキテクチャがすべて含まれています。
GPU が以下の表に含まれていれば変更不要です。

| GPU 世代 | CC 値 | 代表モデル |
|----------|-------|-----------|
| Turing   | sm_75 | GTX 16xx / RTX 20xx |
| Ampere   | sm_80 | A100 |
| Ampere   | sm_86 | RTX 30xx |
| Ada Lovelace | sm_89 | RTX 40xx |

RTX 50xx (sm_100) など上記にないアーキテクチャを使う場合は、
`CMakeLists.txt` の該当行に CC 値を追記してください。

```cmake
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;100")
```

---

## 4. ビルド

プロジェクトルートで以下を実行します。

```powershell
# CMake プリセット使用（推奨）
cmake --preset windows-release
cmake --build --preset windows-release
```

プリセットを使わない場合（パスは自分の環境に合わせて変更）:

```powershell
cmake -B build -S . `
  -DCMAKE_TOOLCHAIN_FILE="<vcpkg のクローン先>/scripts/buildsystems/vcpkg.cmake" `
  -DCMAKE_PREFIX_PATH="<Qt のインストール先>/<バージョン>/msvc2022_64" `
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

---

## 5. 実行

```powershell
.\build\Release\BlackholeSim.exe
```

テクスチャファイルは自動的に exe と同じフォルダへコピーされます。

---

## 操作方法

| 操作 | 動作 |
|------|------|
| 左ドラッグ | 視点回転（軌道） |
| 右ドラッグ | カメラロール |
| ホイール | ズーム |
| 質量 M | 黒穴の質量（シュワルツシルト半径 rs = 2M） |
| スピン a/M | Kerr 回転パラメータ（−1〜1） |
| 電荷 Q/M | Kerr-Newman 電荷（−1〜1） |
| 降着円盤 | 表示 / 非表示トグル |
| 積分ステップ | 精度調整（多いほど重い） |
| PNG 保存 | 現在フレームを画像出力 |
