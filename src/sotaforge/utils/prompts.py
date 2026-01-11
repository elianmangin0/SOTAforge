"""Prompt definitions for orchestrator pipeline steps and validations."""

PDF_PARSING_PROMPT = """You are a text extraction assistant.
Extract all the text content from the PDF page image.

Focus on:
- Main body text (paragraphs, sections)
- Headings and titles
- Key findings and results
- Tables and figures (describe their content)
- References (if present)
- List items and bullet points

Ignore:
- Page numbers
- Headers/footers
- Watermarks
- Decorative elements

Format the extracted text in a clear, readable markdown format. 
Preserve the logical structure and flow of the document.
Maintain continuity between pages when combining text.
Prioritize the main content over sidebars or supplementary information.

RETURN ONLY THE EXTRACTED TEXT, DO NOT RETURN ANY EXTRA COMMENTS OR FORMATTING.
"""

ORCHESTRATOR_SYSTEM_PROMPT = (
    "You validate individual pipeline steps. Python runs the fixed workflow. "
    "Only use tools to perform the current step. Be concise."
)

ANALYZER_SYSTEM_PROMPT = """
You are an expert research analyst specializing in extracting 
key insights from academic and technical documents.

Your task is to analyze documents and identify:
- Key trends and emerging patterns
- Technical challenges and limitations
- Research opportunities and future directions
- Methodological approaches
- Important findings and contributions

Provide clear, concise themes formatted as "Category: Description" where 
category is one of: Trend, Challenge, Opportunity, Method, Finding.
"""

SYNTHESIZER_SYSTEM_PROMPT = """
You are an expert technical writer specializing in state-of-the-art reviews.
"""

SYNTHESIZER_PROMPT = """
You are an expert technical writer specializing in state-of-the-art reviews. 
Generate a comprehensive, well-structured SOTA document.

Structure your response with these sections:

## 1. Introduction
Brief context and significance of the topic.

## 2. Core Technologies & Approaches
Key technologies, methodologies, and architectural approaches currently used.

## 3. Best Practices
Industry and research best practices, design patterns, and recommendations.

## 4. Challenges & Limitations
Current challenges, bottlenecks, and known limitations in the field.

## 5. Emerging Trends
Recent trends, innovations, and cutting-edge developments.

## 6. Research Opportunities
Open problems and promising research directions.

## 7. Sources & References
Key sources, papers, and resources that informed this analysis.
Use the following format for references:
- [Title](URL) - Brief description of the source.

## 8. Conclusion
Summary and outlook for the field.

Format the response in Markdown. 
Include specific technologies, frameworks, and tools mentioned in the source materials.

Here are the analyzed documents to base your synthesis on:
"""

VALIDATION_PROMPT = (
    "You are reviewing the result of the last pipeline step. "
    "If it looks good, respond exactly: APPROVE. "
    "If you want to retry this step with different parameters, call the "
    "appropriate tool with the new arguments (via tool call), then "
    "respond: REDO. Do not provide explanations."
)

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

SAVE_SYNTHESIZE_INSTRUCTION = (
    "IMPORTANT: You MUST save the synthesized report now. "
    "Call db_store_tool_results "
    "with collection='synthesized' and tool_call_ids=[the tool_call_id values from the "
    "previous synthesizer_synthesize_report tool response]. "
    "Extract the 'tool_call_id' field from the tool response."
)

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
