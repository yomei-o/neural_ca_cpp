# 🧬 Neural Cellular Automata (C++ / WASM)

**日本語** | [English](README.en.md)

1個の「種セル」から、**学習した近傍ルール（小さなニューラルネット）だけ**で絵が“成長”し、
一部を壊しても勝手に**再生（自己修復）**する Neural Cellular Automata (NCA) の実装です。

学習エンジン（自作の逆伝播 autograd）から推論（WebAssembly デモ）まで、**すべて依存ライブラリなしの純 C++** で書かれています。

## 🎮 オンラインデモ

絵をクリック／ドラッグで**破壊**すると、数十ステップで**再生**します。

| デモ | 内容 |
|------|------|
| [😀 スマイリー（にこちゃんマーク）](https://yomei-o.github.io/neural_ca/) | コード内蔵のターゲット画像から学習 |
| [🐱 ドラえもん](https://yomei-o.github.io/neural_ca_dora/) | 任意画像（`--target`）から学習した例 |

<p>
  <img src="wasm_grown.png" width="150" alt="育った状態">
  <img src="wasm_damaged.png" width="150" alt="破壊した状態">
  <img src="wasm_healed.png" width="150" alt="自己修復した状態">
</p>

*左：種から成長 → 中央：クリックで破壊 → 右：自己修復*

## 💡 仕組み

- 各セルは **16 次元の状態** を持ちます（先頭 4 = RGBA プリマルチプライド、残り 12 = 隠れチャンネル）。
- **知覚（Perception）**：各チャンネルに対して 固定の Sobel フィルタ（identity / ∂x / ∂y）を畳み込み、近傍の情報を集めます（→ 3×C 次元）。
- **ルール（Rule）**：知覚結果を **小さな 2 層 MLP**（`3C → H → C`, ReLU）に通し、状態の差分 `Δstate` を出力します。
- **更新**：`state += Δstate` を **全セル並列**に、確率的（各セル 50%）で適用。さらに「近傍のアルファが十分あるセルだけ生かす」**alive マスク**を掛けます。
- これを数十ステップ繰り返すと、**局所ルールだけから全体の形が創発**します。

### 自己修復が学習できる理由

学習時に **状態プール（pool）＋ ランダムなダメージ** を使います。

- プールから状態をサンプルし、最悪の個体は種に戻し、良い個体には**わざと円形の破壊を加える**。
- そこから T ステップ走らせてターゲット画像との MSE を最小化。
- これにより「形を**保持**し続ける安定性」と「壊れても**元に戻す**再生能力」を同時に獲得します。

## 🛠 ビルドと学習（CLI）

CPU 上で学習し、学習済みの重み（`nca.bin`）を書き出します。

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# 内蔵のスマイリーを学習
./build/nca --iters 2000 --lr 2e-3 --out nca.bin

# 任意の画像を学習（--target は raw 形式: int w, int h, その後 w*h*4 の float RGBA プリマルチプライド）
./build/nca --iters 2000 --target dora_target.raw --out dora.bin
```

主なオプション：

| オプション | 説明 | 既定値 |
|-----------|------|--------|
| `--n`     | グリッドの一辺 | 28 |
| `--iters` | 学習イテレーション数 | 2000 |
| `--lr`    | 学習率（Adam） | 2e-3 |
| `--h`     | MLP 隠れ層の次元 | 128 |
| `--b`     | バッチサイズ | 4 |
| `--tmax`  | 1 イテレーションの最大ステップ数 | 64 |
| `--target`| ターゲット画像（raw 形式）| （内蔵スマイリー）|
| `--out`   | 出力する重みファイル | nca.bin |

学習中は 200 イテレーションごとに `preview.ppm` を書き出し、終了時に `grown.ppm` / `healed.ppm`（成長→破壊→修復のデモ）を出力します。学習ログの例：

```
iter    1 | T=46 | loss 0.08610
iter   25 | T=43 | loss 0.03983
...
iter 1600 | T=50 | loss 0.00731
```

## 🌐 WASM デモのビルド

学習済みの `nca.bin` を埋め込んで Emscripten でビルドします。

```bash
# EMSDK のパスを環境変数で指定（既定: /c/prog/emsdk/emsdk）
EMSDK=/path/to/emsdk NET=nca.bin OUT=wasmdist ./build_wasm_ca.sh
```

`wasmdist/`（`ca.js` + `ca.wasm` + `index.html`）を任意の静的ホスティングに置けば動きます。

## 📁 ファイル構成

| ファイル | 役割 |
|---------|------|
| `autograd.h` / `autograd.cpp` | 依存なしの逆伝播 autograd エンジン（Tensor, conv2d, relu, Adam ほか）|
| `nca.cpp` | 学習本体（プール＋ダメージ学習、CLI）|
| `wasm_ca.cpp` | 推論専用ランタイム（autograd 不要）。WASM から呼ぶ |
| `build_wasm_ca.sh` | Emscripten ビルドスクリプト |
| `CMakeLists.txt` | CLI 学習バイナリのビルド設定 |
| `wasmdist/` | スマイリーの WASM デモ一式 |
| `wasmdist_dora/` | ドラえもんの WASM デモ一式 |
| `*.bin` | 学習済みの重み |
| `*.png` / `*.ppm` | プレビュー・スクリーンショット |

## 📝 ライセンス / 注意

- コードとスマイリーのターゲット画像はオリジナルです。
- ドラえもんのデモは学習の仕組みを示すための **個人的な学習例（ローカル利用）** です。キャラクター画像の権利は各権利者に帰属します。

## 🔗 参考

- Mordvintsev et al., ["Growing Neural Cellular Automata"](https://distill.pub/2020/growing-ca/) (Distill, 2020) — 本実装が着想を得た元論文。
