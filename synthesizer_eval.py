import re

from config import init_global_settings
from synthesizer_trainset import synthesizer_trainset
from workflow import Synthesizer


def evaluate_synthesizer():
    init_global_settings()

    synthesizer = Synthesizer()
    synthesizer.load("optimized_synthesizer.json")

    for e_idx, gold in enumerate(synthesizer_trainset[:]):
        print(f"\n{'=' * 50}\n[Example {e_idx + 1}] Query: {gold.query}")

        # 執行推論
        pred = synthesizer(query=gold.query, context=gold.context)

        gold_ans = str(gold.answer)
        pred_ans = str(pred.answer)

        # 檢驗引用格式標籤
        has_citation = bool(re.search(r"\[Doc \d+(?:[,;]\s*Doc \d+)*\]", pred_ans))

        print(f"\n[Gold Answer]:\n{gold_ans}")
        print(f"\n[Pred Answer]:\n{pred_ans}")
        print("\n[Validation]:")
        print(f"- Citation Format ([Doc X]): {'Pass' if has_citation else 'Fail'}")


if __name__ == "__main__":
    evaluate_synthesizer()
