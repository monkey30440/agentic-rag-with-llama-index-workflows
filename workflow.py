from datetime import datetime

import dspy
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters
from llama_index.core.workflow import (
    Context,
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.postprocessor.cohere_rerank import CohereRerank
from pydantic import BaseModel, Field

from config import COHERE_API_KEY


class RetrievalTask(BaseModel):
    mode: str = Field(
        ...,
        description="Retrieval mode: 'precision' (specific date/version) or 'global' (historical evolution).",
    )
    target_date: str | None = Field(
        None, description="Target effective date (YYYY-MM-DD) for the task. None if global mode."
    )
    target_version: str | None = Field(
        None, description="Target version number (e.g., '1.0', '4.3.1'). None if applicable to all."
    )
    protocol_type: str | None = Field(
        None, description="Document type: 'Test Protocol' or 'Assessment Protocol'."
    )
    system_domain: str | None = Field(
        None, description="System domain: 'Car-to-Car' or 'Vulnerable Road User'."
    )
    rewritten_query: str = Field(
        ...,
        description="""Optimized search keywords for this task. Must be noun phrases or section titles (e.g., 'AEB system section').
          **STRICTLY EXCLUDE** comparative terms like 'new', 'old', 'changed', 'difference', 'compare'.""",
    )


class RetrievalTaskList(BaseModel):
    tasks: list[RetrievalTask] = Field(
        ..., description="List of decomposed parallel retrieval tasks."
    )


class RetrievalTaskEvent(Event):
    task: RetrievalTask


class RetrievalResultEvent(Event):
    nodes: list[NodeWithScore]
    task: RetrievalTask


class AugmentedContextEvent(Event):
    results: list[RetrievalResultEvent]
    original_query: str


class QueryToTasks(dspy.Signature):
    """
    Role: Euro NCAP Technical Retrieval Planner.
    Objective: Decompose the user's natural language inquiry into precise, atomic retrieval tasks optimized for a vector database containing Euro NCAP protocols.

    1. Retrieval Mode Logic:
       - Set mode="global" ONLY IF the query asks for history, evolution, introduction dates, or broad trends across versions (e.g., "When was AEB introduced?", "History of VRU").
       - Set mode="precision" for specific versions, dates, current standards (Today), or specific technical requirements (e.g., "v4.3 requirements", "test speed in 2023").

    2. Target Version & Date Logic (Only for mode="precision"):
       - target_version: Extract explicit version numbers (e.g., "4.3", "1.0"). If unrelated to a specific version, set to None.
       - target_date: Extract explicit years or dates. If the query implies "current" or "today", use the 'today' input date. Otherwise, set to None.
       - FOR mode="global": ALWAYS set target_version and target_date to None.

    3. Protocol Type Logic (Priority Rule):
       - "Assessment Protocol": CHOOSE ONLY IF the query explicitly mentions scoring, points, stars, ratings, or evaluation criteria.
       - "Test Protocol": DEFAULT SELECTION. Use this for physical testing, scenarios, speeds, tolerances, dummy positioning, and vehicle setup.

    4. System Domain Logic (Strict Mapping):
       - "Car-to-Car (C2C)": If query contains "Car-to-Car", "C2C", or scenarios starting with "CC" (CCRs, CCRm, CCRb, CCFtap, CCCscp, CCFhol, CCFhos).
       - "Vulnerable Road User (VRU)": If query contains "Pedestrian", "Bicyclist", "Motorcyclist", "VRU", or scenarios starting with "CP", "CB", "CM".
       - None: For all other domains (e.g., Safety Assist, LSS, Occupant Status) or if the domain is ambiguous.

    5. Query Rewriting Logic:
       - Optimization: Convert natural language into technical noun phrases or section titles (e.g., "how fast" -> "test speed specification").
       - Filtration: STRICTLY REMOVE all comparative or temporal terms (e.g., "difference", "changed", "new", "old", "vs", "improvement"). The query must focus on the *topic* to be retrieved, not the *action* of comparing.
    """

    query: str = dspy.InputField(desc="User's technical inquiry about Euro NCAP standards.")
    today: str = dspy.InputField(
        desc="Current date (YYYY-MM-DD) to resolve 'current' or 'latest' references."
    )

    plan: RetrievalTaskList = dspy.OutputField(
        desc="A list of atomic retrieval tasks structured according to the logic above."
    )


class ContextToAnswer(dspy.Signature):
    """
    Role: Lead Euro NCAP Technical Auditor.
    Objective: Synthesize retrieved protocol fragments into a factual, comparative, and highly-cited technical response.

    Audit Guidelines:
    1. Cross-Version Synthesis: If context covers multiple versions, organize the answer chronologically or as a 'Before vs. After' comparison.
    2. Fact-Checking: Only state information explicitly supported by the context. If the query asks for "changes", identify the delta between versions.
    3. Missing Data: If the context is empty or irrelevant, state "Insufficient reference material for [Topic]" and explain what is missing.

    Citation Standard:
    - Every claim MUST be followed by a citation in brackets: [File: FileName, Version: X].
    - If a fact is a deduction from two sources, cite both: [File: A; File: B].

    Output Style:
    - Use bullet points for technical requirements.
    - Maintain a formal, neutral, and auditing-style tone.
    """

    context: str = dspy.InputField(desc="Aggregated protocol snippets with metadata")
    query: str = dspy.InputField(desc="User's technical inquiry")

    answer: str = dspy.OutputField(desc="Cited technical report or delta analysis")


class Planner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(QueryToTasks)

    def forward(self, query: str, today: str):
        return self.prog(query=query, today=today)


class Synthesizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(ContextToAnswer)

    def forward(self, context: str, query: str):
        return self.prog(context=context, query=query)


class EuroNCAPWorkflow(Workflow):
    def __init__(self, index: VectorStoreIndex, timeout: int = 60, verbose: bool = True):
        super().__init__(timeout=timeout, verbose=verbose)
        self.index = index

        self.dspy_planner = Planner()
        # self.dspy_planner.load("optimized_planner.json")

        self.dspy_synthesizer = Synthesizer()
        # self.dspy_synthesizer.load("optimized_synthesizer.json")

    @step
    async def planner(self, ctx: Context, ev: StartEvent) -> RetrievalTaskEvent | StopEvent | None:
        print("[Planner] Decomposing query via DSPy...")

        query = ev.query
        today_str = datetime.now().strftime("%Y-%m-%d")

        prediction = self.dspy_planner(query=query, today=today_str)

        plan = prediction.plan

        task_count = len(plan.tasks)

        if task_count == 0:
            print("[Planner] No retrieval tasks generated. Reason: Out of domain knowledge.")
            return StopEvent(result="Sorry, I do not have the knowledge to answer that question.")

        await ctx.store.set("total_tasks", task_count)
        await ctx.store.set("results", [])
        await ctx.store.set("original_query", query)

        if hasattr(prediction, "reasoning"):
            print(f"[Planner Reasoning]: {prediction.reasoning}")

        print(f"[Planner] Decomposition complete. Total tasks: {task_count}")

        for i, t in enumerate(plan.tasks):
            print(
                f"   Task {i + 1}: [{t.mode}] Date={t.target_date} | Version={t.target_version} | Query='{t.rewritten_query}'"
            )
            ctx.send_event(RetrievalTaskEvent(task=t))

        return None

    @step
    async def retriever(self, ev: RetrievalTaskEvent) -> RetrievalResultEvent:
        task = ev.task
        print(f"[Retriever] Processing: {task.rewritten_query} (Mode={task.mode})")

        filters = []
        if task.mode == "precision" and task.target_date:
            target_date_int = int(task.target_date.replace("-", ""))
            filters.extend(
                [
                    MetadataFilter(
                        key="start_date", value=target_date_int, operator=FilterOperator.LTE
                    ),
                    MetadataFilter(
                        key="end_date", value=target_date_int, operator=FilterOperator.GTE
                    ),
                ]
            )

        if task.mode == "precision" and task.target_version:
            filters.append(
                MetadataFilter(key="version", value=task.target_version, operator=FilterOperator.EQ)
            )

        if task.protocol_type:
            filters.append(
                MetadataFilter(
                    key="protocol_type", value=task.protocol_type, operator=FilterOperator.EQ
                )
            )

        metadata_filters = MetadataFilters(filters=filters) if filters else None

        vector_retriever = self.index.as_retriever(
            similarity_top_k=30,
            filters=metadata_filters,
        )
        vector_nodes = await vector_retriever.aretrieve(task.rewritten_query)

        if vector_nodes:
            reranker = CohereRerank(
                top_n=5, model="rerank-multilingual-v3.0", api_key=COHERE_API_KEY
            )
            reranked_nodes = reranker.postprocess_nodes(
                vector_nodes, query_str=task.rewritten_query
            )
            print(f"[Retriever] Reranking complete. Selected top {len(reranked_nodes)} nodes")
        else:
            reranked_nodes = []
            print("[Retriever] No nodes found via vector search.")

        return RetrievalResultEvent(nodes=reranked_nodes, task=task)

    @step
    async def aggregator(
        self, ctx: Context, ev: RetrievalResultEvent
    ) -> AugmentedContextEvent | None:
        total_tasks = await ctx.store.get("total_tasks")
        original_query = await ctx.store.get("original_query")

        results = await ctx.store.get("results", default=[])
        results.append(ev)
        await ctx.store.set("results", results)

        current_count = len(results)
        print(f"[Aggregator] Received result {current_count}/{total_tasks}...")

        if current_count >= total_tasks:
            print("[Aggregator] All results collected. Proceeding to Synthesizer.")

            return AugmentedContextEvent(results=results, original_query=original_query)

        return None

    @step
    async def synthesizer(self, ev: AugmentedContextEvent) -> StopEvent:
        print("[Synthesizer] Generating response via DSPy...")

        context_parts = []
        has_content = False

        for i, result in enumerate(ev.results):
            task_info = result.task
            if not result.nodes:
                continue

            has_content = True
            label = (
                f"Source {i + 1} (Date: {task_info.target_date or 'Any'}, Mode: {task_info.mode})"
            )

            content = ""
            for node in result.nodes:
                file_name = node.node.metadata.get("file_name", "Unknown File")
                content += f"\n[File: {file_name}]\n{node.node.get_content()}\n"
            context_parts.append(f"=== {label} ===\n{content}\n")

        if not has_content:
            return StopEvent(
                result="Sorry, I could not find any relevant information in the provided protocols matching your criteria."
            )

        full_context = "\n".join(context_parts)

        prediction = self.dspy_synthesizer(context=full_context, query=ev.original_query)

        print(f"[Synthesizer Reasoning]: {prediction.reasoning}")

        print("[Synthesizer] Response generated.")

        return StopEvent(result=str(prediction.answer))
