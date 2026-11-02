# 実装課題：架空の自作ゲーム体験会 FAQ

元ノートブックの `FAQ` / `GROUNDED` / `UNGROUNDED` を独自の架空題材に置き換えた。実在のイベント情報ではない。

- [解答と考察](../answer.md)
- [再実行用ノートブック](../../../notebooks/lora_game_event_faq.ipynb)
- [独自FAQ・質問の定義](faq_domain.py)
- `results/`：Colabで生成されたデータ、前後の生成原文、採点結果、学習ログ

## 再実行

1. `notebooks/lora_game_event_faq.ipynb` をGoogle Colabで開き、T4 GPUを選ぶ。
2. 上からセクション7まで実行する。依存ライブラリとモデルrevisionはノートブックの固定値を使う。DriveのマウントやAPIキーは不要。
3. `/content/aie_day2_game_faq_results.zip` を保存する。学習済みadapterは `/content/aie_day2_game_faq/adapter/` に保存される。
4. 終了時に「ランタイムを接続解除して削除」を実行する。

結果ファイルを `results/` に展開すると、リポジトリのルートから次で原文の再採点と整合性確認ができる（標準ライブラリのみ、モデル実行なし）。

```sh
python3 lectures/02/implementation/verify_results.py
```

## 比較条件と限界

train 14件、valid 7件、test 21件の質問は重複しない。同じ5項目のFAQを各入力に含めるため、この実験は未知の知識の暗記能力ではなく、与えられたFAQに基づく回答形式・判断を測る。testは根拠あり15問、根拠なし6問。未知情報の追加プローブは3問と、FAQに記載のある対照1問。

モデル・revision・プロンプト・評価質問・生成条件は学習前後で揃えた。セクション6の学習前プローブは `disable_adapter()` でadapterを無効にする。validはTrainerに渡しているが、講義の元設定どおり定期評価や早期終了には使っていない。学習は1回で、testに合わせたハイパーパラメータ探索はしていない。

内容の自動採点は `needs_review` の一致と必須語句の部分文字列一致であり、意味全体の正しさを保証しない。全21問の原文も確認し、解答では自動採点と文面の評価を区別する。1ドメイン・小標本の結果を一般的な安全性や知識能力に拡張しない。

データ作成・コード変更・Colab操作・考察の草稿にはCodexを使用した。実行の有無と数字はColabの生成物に基づく。
