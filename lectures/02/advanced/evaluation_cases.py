"""学習前に固定する42問。用語の定義を問う36問と用語集未掲載の6問。"""
CASES = {
    "lora": {
        "alias": "Low-Rank Adaptation",
        "paraphrase": "学習済みの重みを固定し、小さな二つの行列だけを調整する方法を説明してください。",
        "groups": [["固定", "凍結"], ["行列"], ["低ランク", "2つ", "二つ", "小さな"]],
    },
    "qlora": {
        "alias": "Quantized LoRA",
        "paraphrase": "元モデルを4ビットで保持して省メモリで追加の行列を学習する手法は何ですか。",
        "groups": [["量子化", "4ビット", "4bit"], ["LoRA"], ["メモリ", "GPU"]],
    },
    "sft": {
        "alias": "Supervised Fine-Tuning",
        "paraphrase": "質問と模範回答の組を与えて、指示への従い方を身につけさせる学習方法を説明してください。",
        "groups": [["入力", "質問"], ["出力", "回答"], ["ペア", "組", "教師", "正解"]],
    },
    "peft": {
        "alias": "Parameter-Efficient Fine-Tuning",
        "paraphrase": "モデル全体を更新せず、一部または追加部分だけを学習する手法の総称を説明してください。",
        "groups": [["一部", "追加", "小さな"], ["パラメータ"], ["学習"]],
    },
    "overfitting": {
        "alias": "overfitting",
        "paraphrase": "訓練例にはよく合うのに、未見の入力では性能が落ちる状態を説明してください。",
        "groups": [["学習データ", "訓練"], ["未知", "未見", "汎化", "検証"], ["適合", "性能", "過度"]],
    },
    "context-window": {
        "alias": "context window",
        "paraphrase": "モデルが一回の呼び出しで処理できる、入力と出力を合わせた長さの制約を説明してください。",
        "groups": [["入力"], ["出力"], ["トークン"], ["上限", "最大", "限界"]],
    },
    "hallucination": {
        "alias": "hallucination",
        "paraphrase": "存在しない情報を事実らしい文章として作ってしまう現象と、抑える方法を説明してください。",
        "groups": [["事実", "存在しない", "誤"], ["生成", "出力", "作"], ["参照", "根拠", "評価", "外部"]],
    },
    "agent-loop": {
        "alias": "agent loop",
        "paraphrase": "関数の実行結果をモデルへ戻し、最終回答まで次の行動を繰り返す処理を説明してください。",
        "groups": [["ツール"], ["結果"], ["モデル", "LLM"], ["繰り返", "ループ"]],
    },
    "homework-levels": {
        "alias": "宿題レベル",
        "paraphrase": "AIE2026では宿題の難易度はどう分かれ、どこまで取り組めばその回の完了になりますか。",
        "groups": [["基礎"], ["実装"], ["発展"], ["1つ", "一つ", "1段階", "いずれか", "どれか"]],
    },
    "answer-md": {
        "alias": "解答ファイル",
        "paraphrase": "AIE2026で各回の演習結果はどこへ記録し、どのような変更依頼として提出しますか。",
        "groups": [["answer.md"], ["lectures"], ["PR", "プルリクエスト"], ["fork", "フォーク"], ["main"]],
    },
    "lec-start-tag": {
        "alias": "開始タグ",
        "paraphrase": "前の回を終えていない受講者が、AIE2026の各回の開始時点のコードから作業するための目印は何ですか。",
        "groups": [["タグ"], ["開始"], ["lecNN-start", "lec03-start"]],
    },
    "default-model": {
        "alias": "デフォルトモデル",
        "paraphrase": "AIE2026の標準のモデル名と、それを切り替えるための環境変数名を教えてください。",
        "groups": [["gemini-3.5-flash-lite"], ["AIE_MODEL"]],
    },
}

# 実在の一般用語が未掲載でも、その用語の説明が事実誤認とは限らない。
# ここでは「講座の用語集に基づく回答」という対象範囲の保留判断を測る。
UNKNOWN = [
    ("バッチ正規化", "actual_out_of_scope"),
    ("ドロップアウト", "actual_out_of_scope"),
    ("Adam", "actual_out_of_scope"),
    ("量子ロラ同期器", "invented_name"),
    ("ネオトークン同期器", "invented_near_match"),
    ("プロンプト超越変換器", "invented_near_match"),
]


def make_cases(terms):
    by_id = {t["id"]: t for t in terms}
    rows = []
    for term_id, spec in CASES.items():
        term = by_id[term_id]
        assert spec["alias"] in term["aliases"]
        for kind, question in [
            ("canonical", f"用語集の『{term['term']}』の定義を教えてください。"),
            ("alias", f"{spec['alias']}という用語は、講座ではどういう意味ですか。"),
            ("paraphrase", spec["paraphrase"]),
        ]:
            rows.append({"id": f"{term_id}:{kind}", "kind": kind, "term_id": term_id,
                         "question": question, "expected_definition": term["definition"],
                         "groups": spec["groups"], "expect_abstention": False})
    for name, subtype in UNKNOWN:
        rows.append({"id": f"unknown:{name}", "kind": "unknown", "subtype": subtype,
                     "term_id": None, "question": f"用語集の『{name}』の定義を教えてください。",
                     "expected_definition": None, "groups": [], "expect_abstention": True})
    return rows
