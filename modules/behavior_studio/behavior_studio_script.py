from groq import Groq
from configs.settings import settings


def optimized_prompt(raw_config_text: str) -> str | None:
    if not raw_config_text.strip():
        return None

    meta_prompt = f"""
You are a senior prompt engineer specializing in designing high-quality system prompts for domain-based audience RAG chatbots.

Your task is to transform the raw chatbot configuration below into ONE optimized system prompt.

The resulting prompt must be clear, structured, and optimized for reliable LLM behavior.

Design principles:
- Clear role definition
- Deterministic behavior
- Strong hallucination prevention
- Efficient RAG usage
- Concise instructions for fast inference (Groq optimized)

The final system prompt should be structured in the following order:

1. ROLE
Define what the assistant is and its domain expertise.

2. PERSONALITY
Define tone, language, and communication style.

3. KNOWLEDGE USAGE (RAG RULES)
Explain how the assistant should use retrieved knowledge:
- Prefer information from retrieved context
- Do not fabricate information
- If context does not contain the answer, follow fallback instructions

4. BEHAVIOR RULES
Include domain rules, audience awareness, and restrictions.

5. SAFETY & POLICY
Apply provided policy restrictions if present.

6. RESPONSE FORMAT
Define formatting rules: concise answers, no duplicate ideas.

7. FALLBACK LOGIC
If a fallback_message exists, clearly instruct when to use it.

8. AVOIDING REDUNDANT INSTRUCTIONS
If configuration fields present in the additional instructions present, do not repeat it.

Constraints:
- Maximum ~450 tokens
- No unnecessary verbosity
- No invented information
- Do not repeat the configuration text verbatim
- Use direct instructions ("You are...", "Always...", "Never...")

Output Rules:
- Output ONLY the final system prompt
- No explanations
- No markdown fences
- No commentary

Raw chatbot configuration:

{raw_config_text}

Final system prompt:
"""
    return _stream_chat(meta_prompt)


def generate_prompt_suggestions(prompt: str) -> str | None:
    """
    Generate list of prompt improvement suggestions
    """
    if not prompt.strip():
        return None

    meta_prompt = f"""
You are a senior prompt engineer especiallizing in generating suggestion improvements based on the prompt.  

Your task is to analyze the system prompt below and suggest small improvements that could make it better.

Focus on improving:
- clarity
- tone
- response behavior
- safety rules
- hallucination prevention
- RAG usage
- response formatting

Rules:
- Generate up to 4 suggestions
- Suggestions must be short (3–8 words)
- Each suggestion describes ONE improvement from the prompt
- Do NOT explain suggestions
- Do NOT repeat the prompt
- Do NOT explain changes
- Do NOT include commentary
- Do NOT include markdown formatting

Output format (STRICT):

Return a valid JSON array of strings.

Examples:

With suggestions:
["Improve fallback clarity", "Add tone constraints"]

No suggestions:
[]

Prompt to analyze:
{prompt}

Suggestions:
"""

    return _stream_chat(meta_prompt)


def improve_prompt_based_suggestions(prompt: str, suggestions: str) -> str | None:
    """Response goal: current prompt + improved prompt (based on suggestions)"""
    if not prompt or not suggestions:
        return None

    meta_prompt = f"""
You are a senior prompt engineer specializing in refining system prompts for production AI assistants.

Your task is to improve the CURRENT PROMPT by applying the provided SUGGESTED IMPROVEMENTS.

Important rules:
- Preserve the original intent and meaning of the prompt
- Apply the suggestions where appropriate
- Do NOT remove useful instructions unless the suggestion implies improvement
- Do NOT invent new requirements beyond the suggestions
- Keep the prompt clear, concise, and structured
- Prefer direct instructions such as "You are...", "Always...", "Never..."

How to apply suggestions:
- Integrate them naturally into the prompt
- If multiple suggestions relate to the same concept, combine them cleanly
- Avoid duplicating instructions
- Maintain logical order of sections

Output requirements:
- Return ONLY the improved prompt
- Do NOT explain changes
- Do NOT include commentary
- Do NOT include markdown formatting
- Do NOT include the suggestion list in the output
- Keep the prompt concise (prefer under ~450 tokens).

CURRENT PROMPT:
{prompt}

SUGGESTED IMPROVEMENTS:
{suggestions}

Improved prompt:
"""

    return _stream_chat(meta_prompt)


def improve_current_prompt(prompt: str) -> str | None:
    """
    Regenerate prompt.
    Extract the prompt text (whether it's already improved or not)
    and produce a cleaner, more optimized version while preserving intent.
    """

    if not prompt:
        return None

    meta_prompt = f"""
You are a senior prompt engineer specializing in optimizing system prompts for production AI assistants.

Your task is to improve the prompt below.

Goals:
- Preserve the original meaning and intent
- Improve clarity and structure
- Remove redundant or unclear instructions
- Ensure instructions are concise and deterministic
- Use clear directive language such as "You are...", "Always...", "Never..."

Guidelines:
- Do NOT invent new features or rules
- Do NOT significantly change the behavior of the prompt
- Do NOT add unnecessary verbosity
- Merge duplicate instructions
- Organize instructions logically if possible

Optimization priorities:
1. clarity
2. structure
3. conciseness
4. reliability for LLM behavior

Output requirements:
- Return ONLY the regenerated prompt
- No explanations
- No commentary
- No markdown formatting
- Do not repeat the input prompt verbatim unless necessary
- Keep the prompt concise (prefer under ~450 tokens).

Prompt to regenerate:
{prompt}

Regenerated prompt:
"""

    return _stream_chat(meta_prompt)
    
    
def simplify_current_prompt(prompt: str) -> str | None:
    """Simplify the current prompt"""

    if not prompt or not prompt.strip():
        return None
    
    meta_prompt = f"""
You are a senior prompt engineer specializing in optimizing prompts for production AI systems.

Your task is to simplify the prompt below while preserving its original meaning and behavior.

Goals:
- Reduce unnecessary verbosity
- Remove redundant or repeated instructions
- Merge similar rules where possible
- Shorten long or complex sentences
- Keep instructions clear and direct

Important constraints:
- Do NOT remove important behavioral rules
- Do NOT invent new instructions
- Do NOT change the intent of the prompt
- Preserve all essential guidance and restrictions

Optimization priorities:
1. clarity
2. conciseness
3. deterministic instructions
4. efficient token usage

Keep the simplified prompt significantly shorter if possible while maintaining the same functionality.

Output requirements:
- Return ONLY the simplified prompt
- No explanations
- No commentary
- No markdown formatting

Prompt to simplify:
{prompt}

Simplified prompt:
"""

    return _stream_chat(meta_prompt)
    

def _stream_chat(meta_prompt: str) -> str | None:
    llm = Groq(api_key=settings.LLM_API_KEY)
    
    llm_request = llm.chat.completions.create(
        model=settings.LLM_NAME,
        messages=[{"role": "user", "content": meta_prompt}],
        temperature=0.40,
        max_tokens=1024,
        stream=True
    )
    
    # concat chunks as they arrive
    response = ""
    for reply in llm_request:
        if reply.choices[0].delta.content:
            response += reply.choices[0].delta.content
            
    return response or None
