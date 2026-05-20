# Project Instructions: LOCAL_LLM_TEST2

## Language Policy
- **Primary Language:** Japanese (日本語)
- **Constraint:** All code comments, UI strings, and system messages within the codebase must be written in **Japanese**.
- **Exception:** Technical identifiers (variable names, functions) remain in English as per standard conventions.

## Architecture & Conventions
- **LLM Engine:** Lemonade LLM Server (Exclusively used)
- **Model:** `gemma4-it-e2b-FLM`
- **Streaming:** Final LLM responses must be streamed using `astream`.
- **Multimodal:** Supports image and file uploads via Chainlit.
