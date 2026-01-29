from datetime import datetime

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
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
    rewritten_query: str = Field(
        ...,
        description="""Optimized search keywords for this task. Must be noun phrases or section titles (e.g., 'AEB system section').
          **STRICTLY EXCLUDE** comparative terms like 'new', 'old', 'changed', 'difference', 'compare'.""",
    )


class PlannerOutput(BaseModel):
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


class EuroNCAPWorkflow(Workflow):
    def __init__(self, index: VectorStoreIndex, timeout: int = 60, verbose: bool = True):
        super().__init__(timeout=timeout, verbose=verbose)
        self.index = index
        self.llm = Settings.llm

    @step
    async def planner(self, ctx: Context, ev: StartEvent) -> RetrievalTaskEvent | None:
        print("[Planner] Decomposing query...")

        query = ev.query
        today_str = datetime.now().strftime("%Y-%m-%d")

        prompt_template = PromptTemplate(
            """
            Today is {today}.
            You are a Senior Researcher specializing in Euro NCAP protocols.
            Your task is to decompose complex user queries into specific, independent "Retrieval Tasks".
            This is similar to decomposing a large question into sub-questions for separate retrieval and later aggregation.

            User Query: {query}

            ### Decomposition Strategy:

            1. **Comparative Queries**:
               - If the query involves "Compare A and B" or "Changes from v1 to v2":
               - **Strategy**: Must split into two independent tasks.
               - Task 1: Query the topic for version A (target_version=A).
               - Task 2: Query the topic for version B (target_version=B).
               - *Keyword Tip*: Normalize `rewritten_query` for both tasks to the "Subject Noun" (e.g., set both to "AEB scoring criteria"), do NOT include "compare" or "difference".

            2. **Multi-aspect Queries**:
               - If the query asks for "Definition of X" and "Test speed of Y":
               - **Strategy**: Split into two tasks, querying X and Y respectively.

            3. **Specific Queries**:
               - If the query is simple, generate a single precise task.

            ### Field Filling Rules:
            - **rewritten_query**: Core search terms. Should be "Noun Phrases" or "Section Titles". **STRICTLY FORBID** comparative adjectives (new, changed, diff).
            - **target_version**: Fill strictly if a specific version (e.g., v4.3.1) is targeted. Leave blank for historical lookup.
            - **mode**: Use 'precision' for specific versions, 'global' for cross-time evolution.

            Output JSON format conforming to PlannerOutput.
            """
        )

        plan = await self.llm.astructured_predict(
            PlannerOutput, prompt_template, today=today_str, query=query
        )

        task_count = len(plan.tasks)
        await ctx.store.set("total_tasks", task_count)
        await ctx.store.set("results", [])
        await ctx.store.set("original_query", query)

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
            similarity_top_k=50,
            filters=metadata_filters,
        )
        vector_nodes = await vector_retriever.aretrieve(task.rewritten_query)

        if vector_nodes:
            reranker = CohereRerank(
                top_n=20, model="rerank-multilingual-v3.0", api_key=COHERE_API_KEY
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
    async def synthesizer(self, ev: AugmentedContextEvent) -> StopEvent:  # ctx: Context,
        print("[Synthesizer] Generating response...")

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

        prompt_template = PromptTemplate(
            """
            You are an expert assistant for Euro NCAP. Answer strictly based on the reference materials.

            === Reference Materials Start ===
            {context_str}
            === Reference Materials End ===

            User Query: {query}

            Guidelines:
            1. **Cite Specifics**: Mention the "File Name" or "Version" when citing rules.
            2. **No Hallucination**: If the answer isn't in the text, say "Insufficient reference material".
            3. **Comparative Analysis**: If comparing versions, explicitly highlight the differences found in the text.

            Answer:
            """
        )

        response = await self.llm.acomplete(
            prompt_template.format(context_str=full_context, query=ev.original_query)
        )

        print("[Synthesizer] Response generated.")

        return StopEvent(result=str(response))
