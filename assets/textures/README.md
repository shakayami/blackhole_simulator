# 背景テクスチャ（天球マップ）

このディレクトリの画像はビルド成果物の隣にコピーされ、ブラックホール周辺の背景（天球）として使われます（`../../CMakeLists.txt` 参照）。

## 現在ここに置いているファイル

- `starmap_2020_4k_gal.exr` / `starmap_2020_8k_gal.exr`
- `starmap_2020_4k_gal_print.jpg`
- `starmap_2020_8k_gal.png`
- `celestial_grid_equirectangular_8192x4096.png`（現状コード内では未参照）

## なぜ git 管理外か

- サイズが大きい（`starmap_2020_8k_gal.exr` で約160MB、GitHubの単一ファイル100MB制限を超える）
- NASAのサイトから取得したものだが、**再配布（このリポジトリ経由で第三者に配る形）の際にライセンス上問題がないか未確認**。確認が取れるまでは意図的にコミットしない

`.gitignore` で `assets/textures/` ごと除外している。取得元URLや利用条件が確認できたら、このREADMEに追記し、必要ならファイル単位でgit管理に戻す。

## セットアップ（このディレクトリが空の場合）

1. NASA Scientific Visualization Studio の "Deep Star Maps" 等から該当ファイルをダウンロード
2. このディレクトリ（`assets/textures/`）に配置
3. CMake再構成 → ビルド（存在するファイルだけ自動でコピーされる。何も置かなければアプリはプロシージャル生成の星空にフォールバックする）
