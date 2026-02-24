import os
import re

import dspy

from config import LLM_MODEL, OPENAI_API_KEY, TEMPERATURE
from synthesizer_trainset import synthesizer_trainset
from workflow import Synthesizer

LM = dspy.LM(model=f"openai/{LLM_MODEL}", temperature=TEMPERATURE, api_key=OPENAI_API_KEY)
dspy.configure(lm=LM)

JUDGE_LM = dspy.LM(model="gpt-4o-mini", temperature=0.0, api_key=OPENAI_API_KEY)
TEACHER_LM = dspy.LM(model="gpt-4o-mini", temperature=0.0, api_key=OPENAI_API_KEY)


class SynthesizerFactJudge(dspy.Signature):
    """
    Role: Technical Content Auditor.
    Objective: Evaluate if 'pred_answer' contains the exact same technical facts as 'gold_answer' without hallucination.

    Grading Rubric:
    1.0: Perfect match in facts and numerical values.
    0.5: Missing some minor details but core facts are correct.
    0.0: Contains contradictory numbers, hallucinations, or missing key facts.
    """

    gold_answer: str = dspy.InputField(desc="Ground truth answer containing correct facts.")
    pred_answer: str = dspy.InputField(desc="Model generated answer.")
    rating: float = dspy.OutputField(desc="Float score between 0.0 and 1.0.")


judge = dspy.ChainOfThought(SynthesizerFactJudge)


def synthesizer_metric(gold, pred, trace=None):
    pred_ans = str(pred.answer)
    task_score = 1.0

    # format check
    if not re.search(r"\[Doc \d+\]", pred_ans):
        return 0.0

    # fact check
    with dspy.context(lm=JUDGE_LM):
        res = judge(gold_answer=gold.answer, pred_answer=pred_ans)
    fact_score = float(res.rating)
    task_score *= fact_score

    return task_score


synthesizer_optimizer = dspy.BootstrapFewShotWithRandomSearch(
    metric=synthesizer_metric,
    teacher_settings=dict(lm=TEACHER_LM),
    max_bootstrapped_demos=2,
    num_candidate_programs=3,
    num_threads=1,
)

optimized_synthesizer = synthesizer_optimizer.compile(Synthesizer(), trainset=synthesizer_trainset)
optimized_synthesizer.save("optimized_synthesizer.json")
print("Optimization complete. Optimized synthesizer saved to optimized_synthesizer.json")
