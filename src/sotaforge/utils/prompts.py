"""Prompt definitions for orchestrator pipeline steps and validations."""

# System prompt guiding the orchestrator's role
ORCHESTRATOR_SYSTEM_PROMPT = (
    "You validate individual pipeline steps. Python runs the fixed workflow. "
    "Only use tools to perform the current step. Be concise."
)

# Validation prompt used for each step
# Instruct the LLM to either APPROVE or redo by calling a tool.
VALIDATION_PROMPT = (
    "You are reviewing the result of the last pipeline step. "
    "If it looks good, respond exactly: APPROVE. "
    "If you want to retry this step with different parameters, call the "
    "appropriate tool with the new arguments (via tool call), then "
    "respond: REDO. Do not provide explanations."
)

# Step instructions
SEARCH_INSTRUCTION = (
    "Perform searches to find relevant documents. Research topic: {topic}. "
    "Use both web and academic paper search tools."
)

FILTER_INSTRUCTION = (
    "Filter the search results to high-quality, relevant sources. "
    "First, determine exactly 5 criteria to judge document quality and "
    "relevance for the topic. "
    "Then call filter_filter_results with: "
    "query='{topic}', collection='raw', criteria=[your 5 criteria as strings]. "
    "The filter tool will retrieve documents from the 'raw' collection, "
    "score each on the 5 criteria (1-5), "
    "and keep only documents with mean score > 2."
)

PARSE_INSTRUCTION = (
    "Parse the filtered documents to extract full text content. "
    "Call parser_parse_documents with document_to_process_collection='filtered' "
    "and document_processed_collection='parsed'. "
    "The parser will retrieve all filtered documents, extract text from "
    "web pages and PDFs, store parsed docs into 'parsed', and return a trimmed summary."
)

ANALYZE_INSTRUCTION = (
    "Analyze the parsed content to derive findings, compare methods, "
    "extract metrics, and identify the current SOTA. "
    "Call analyzer_analyze_documents with document_to_process_collection='parsed' "
    "and document_processed_collection='analyzed'. "
    "The analyzer will retrieve parsed documents, enrich them, store into 'analyzed', "
    "and return a trimmed summary."
)

SYNTHESIZE_INSTRUCTION = (
    "Synthesize a concise SOTA report from the analyzed results. "
    "Call write_sota with collection='analyzed'. "
    "The synthesizer will retrieve all analyzed documents, summarize key papers, "
    "methods, benchmarks, metrics, trends, and open gaps."
)

# Per-step save instructions (update Chroma collections)
SAVE_SEARCH_INSTRUCTION = (
    "IMPORTANT: You MUST save the search results now. "
    "Call db_store_tool_results "
    "with collection='raw' and tool_call_ids=[the tool_call_id values you "
    "see in the tool response content from the previous search_search_web "
    "and search_search_papers calls]. "
    "Extract the 'tool_call_id' field from each tool response."
)

SAVE_FILTER_INSTRUCTION = (
    "IMPORTANT: You MUST save the filtered results now. "
    "Call db_store_tool_results "
    "with collection='filtered' and tool_call_ids=[the tool_call_id values from the "
    "previous filter_filter_results tool response]. "
    "Extract the 'tool_call_id' field from the tool response."
)

SAVE_PARSE_INSTRUCTION = (
    "IMPORTANT: You MUST save the parsed documents now. "
    "Call db_store_tool_results "
    "with collection='parsed' and tool_call_ids=[the tool_call_id values from the "
    "previous parser_parse_documents tool response]. "
    "Extract the 'tool_call_id' field from the tool response."
)

SAVE_ANALYZE_INSTRUCTION = (
    "IMPORTANT: You MUST save the analyzed results now. "
    "Call db_store_tool_results "
    "with collection='analyzed' and tool_call_ids=[the tool_call_id values from the "
    "previous analyzer_analyze_documents tool response]. "
    "Extract the 'tool_call_id' field from the tool response."
)

SAVE_SYNTHESIZE_INSTRUCTION = (
    "IMPORTANT: You MUST save the synthesized report now. "
    "Call db_store_tool_results "
    "with collection='synthesized' and tool_call_ids=[the tool_call_id values from the "
    "previous synthesizer_synthesize_report tool response]. "
    "Extract the 'tool_call_id' field from the tool response."
)

# Validation prompts per step
VALIDATE_SEARCH = (
    "Approve the search results and confirm the 'raw' collection is updated "
    "with all search results (web and papers)."
)
VALIDATE_FILTER = (
    "Approve the filtered sources and confirm the 'filtered' collection is "
    "updated with only relevant, high-quality items."
)
VALIDATE_PARSE = (
    "Approve the parsed content and confirm the 'parsed' collection is "
    "updated with extracted full text documents."
)
VALIDATE_ANALYZE = (
    "Approve the analysis results and confirm the 'analyzed' collection is "
    "updated with findings and metrics."
)
VALIDATE_SYNTHESIZE = (
    "Approve the synthesized SOTA report and confirm the 'synthesized' collection is "
    "updated with the final report."
)
