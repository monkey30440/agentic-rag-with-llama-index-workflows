from config import init_global_settings
from planner_trainset import planner_trainset
from workflow import Planner


def evaluate_planner():
    init_global_settings()

    planner = Planner()
    # planner.load("optimized_planner.json")

    for e_idx, gold in enumerate(planner_trainset):
        print(f"[Example {e_idx + 1}] Query: {gold.query}")

        pred = planner(query=gold.query, today=gold.today)

        gold_tasks = gold.plan.tasks
        pred_tasks = pred.plan.tasks

        sorted_gold_tasks = sorted(gold_tasks, key=lambda x: str(x))
        sorted_pred_tasks = sorted(pred_tasks, key=lambda x: str(x))

        for t_idx, (gold_task, pred_task) in enumerate(
            zip(sorted_gold_tasks, sorted_pred_tasks, strict=False)
        ):
            if (
                gold_task.mode != pred_task.mode
                or gold_task.target_date != pred_task.target_date
                or gold_task.target_version != pred_task.target_version
                or gold_task.protocol_type != pred_task.protocol_type
                or gold_task.system_domain != pred_task.system_domain
                # or gold_task.rewritten_query != pred_task.rewritten_query
            ):
                print(
                    f"[Task {t_idx + 1}]\n"
                    f"Gold:\n"
                    f"- Mode: {gold_task.mode}\n"
                    f"- Date: {gold_task.target_date}\n"
                    f"- Version: {gold_task.target_version}\n"
                    f"- Protocol: {gold_task.protocol_type}\n"
                    f"- Domain: {gold_task.system_domain}\n"
                    f"- Query: {gold_task.rewritten_query}\n"
                    f"Pred:\n"
                    f"- Mode: {pred_task.mode}\n"
                    f"- Date: {pred_task.target_date}\n"
                    f"- Version: {pred_task.target_version}\n"
                    f"- Protocol: {pred_task.protocol_type}\n"
                    f"- Domain: {pred_task.system_domain}\n"
                    f"- Query: {pred_task.rewritten_query}"
                )


if __name__ == "__main__":
    evaluate_planner()
