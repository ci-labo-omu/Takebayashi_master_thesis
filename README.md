# 実験の注意点

## 環境設定
- **Edit Configuration:**  
  - `cifar100` の `crop_size` は `32`  
  - `tiny` の場合は `64`  

## コード概要
### `main_pretrain_CL.py`
- タスクインクリメンタルで回せるメインコード。

## `wandb` の設定
- `l:237` の `wandb` の設定:
  - `name=` 実行結果の名前。
  - `project=` 親ファイル。

## タスク追加の設定
- `l:319` で追加する数を記入。
  - **現状は `50` に設定**。

## `base.py` の設定
- `l:250` で初期のコードワード数を決定。
- `l:1556` の `if num_epoch = 199` 以下の処理：
  - `CA+` の適用。
  - `t-SNE` の可視化、コードワード数の Excel への書き出し。
  - `clustering_flag` の `true` / `false` で `CA` の適用を制御。
