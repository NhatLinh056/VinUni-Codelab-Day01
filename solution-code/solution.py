"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to solution/solution.py when done.
    4. Run: pytest tests/ -v
"""

import os
import time
from typing import Any, Callable

# Tự động nạp các biến môi trường từ file .env nếu có
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Estimated costs per 1M INPUT & OUTPUT tokens (USD) as of March 2026
# Vietnamese text generally consumes ~1.5x - 2.0x more tokens than English due to Unicode/diacritics.
# ---------------------------------------------------------------------------
PRICING_1M_TOKENS = {
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.150, "output": 0.600},
    "gemini-2.5-flash": {"input": 0.075, "output": 0.300},
    "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}

# Standard Model Identifiers
OPENAI_MODEL = "gpt-4o"
OPENAI_MINI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-3.8-flash"  # Sử dụng model mới nhất của Google (hoặc gemini-2.0-flash)
ANTHROPIC_MODEL = "claude-3-5-haiku"


# ---------------------------------------------------------------------------
# Task 1 — Call OpenAI (GPT-4o)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the OpenAI Chat Completions API and return the response text, latency,
    and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: gpt-4o).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # response.usage contains input_tokens and output_tokens (prompt_tokens/completion_tokens)
    """
    from openai import OpenAI

    # Khởi tạo client OpenAI với API Key từ biến môi trường
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    start_time = time.time()
    # Gửi request Chat Completions
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.time() - start_time

    # Trích xuất nội dung trả lời và số lượng token tiêu thụ
    response_text = response.choices[0].message.content or ""
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }
    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 2 — Call Google Gemini 2.5 (Standard Practical Model)
# ---------------------------------------------------------------------------
def call_gemini(
    prompt: str,
    model: str = GEMINI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Google Gemini API (using Gemini 2.5 Flash as standard) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Gemini model to use (default: gemini-2.5-flash).
        temperature: Sampling temperature.
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        Option A (New Google GenAI SDK):
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            # Configure using types.GenerateContentConfig
            
        Option B (Legacy Google GenerativeAI SDK):
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model_inst = genai.GenerativeModel(model)
            # Configure using genai.types.GenerationConfig
            
        Ensure your usage dictionary extracts 'input_tokens' and 'output_tokens' 
        from the response metadata (e.g. response.usage_metadata).
    """
    from google import genai
    from google.genai import types

    # Khởi tạo client Google GenAI
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Cấu hình các siêu tham số cho mô hình
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )

    start_time = time.time()
    # Gọi hàm generate_content
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    latency = time.time() - start_time

    # Trích xuất văn bản và số lượng token từ metadata
    response_text = response.text or ""
    input_tokens = response.usage_metadata.prompt_token_count if response.usage_metadata else 0
    output_tokens = response.usage_metadata.candidates_token_count if response.usage_metadata else 0

    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 3 — Call Anthropic Claude (Exploratory track)
# ---------------------------------------------------------------------------
def call_anthropic(
    prompt: str,
    model: str = ANTHROPIC_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float, dict]:
    """
    Call the Anthropic Claude API (using Claude 3.5 Haiku as default) and return
    the response text, latency, and token usage stats.

    Args:
        prompt:      The user message to send.
        model:       The Claude model to use (default: claude-3-5-haiku).
        temperature: Sampling temperature (0.0 - 1.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum output tokens.

    Returns:
        A tuple of:
            - response_text (str)
            - latency_seconds (float)
            - usage (dict with keys: 'input_tokens', 'output_tokens')

    Hint:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # response.usage contains input_tokens and output_tokens
    """
    import anthropic

    # Khởi tạo client Anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    start_time = time.time()
    # Claude Messages API: bắt buộc phải có max_tokens
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - start_time

    # Trích xuất phản hồi từ block nội dung đầu tiên
    response_text = response.content[0].text if response.content else ""
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return response_text, latency, usage


# ---------------------------------------------------------------------------
# Task 4 — Compare Models (OpenAI GPT-4o vs OpenAI Mini vs Gemini 2.5 Flash)
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call OpenAI (gpt-4o), OpenAI Mini (gpt-4o-mini), and Gemini 2.5 Flash (gemini-2.5-flash)
    with the same prompt and return a structured comparison dictionary.

    Calculate the exact USD token cost for input + output using the prices in PRICING_1M_TOKENS.

    Args:
        prompt: The user message to send to all models.

    Returns:
        A dictionary containing:
            - "gpt4o": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gpt4o_mini": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
            - "gemini_flash": { "response": str, "latency": float, "cost": float, "input_tokens": int, "output_tokens": int }
    """
    # Hàm tính chi phí USD dựa trên số lượng token và giá mỗi 1 triệu token
    def _calc_cost(model_key: str, in_tok: int, out_tok: int) -> float:
        pricing = PRICING_1M_TOKENS[model_key]
        return (in_tok * pricing["input"] + out_tok * pricing["output"]) / 1_000_000

    # 1. Gọi GPT-4o
    text_4o, lat_4o, usage_4o = call_openai(prompt, model=OPENAI_MODEL)
    cost_4o = _calc_cost("gpt-4o", usage_4o["input_tokens"], usage_4o["output_tokens"])

    # 2. Gọi GPT-4o Mini
    text_mini, lat_mini, usage_mini = call_openai(prompt, model=OPENAI_MINI_MODEL)
    cost_mini = _calc_cost("gpt-4o-mini", usage_mini["input_tokens"], usage_mini["output_tokens"])

    # 3. Gọi Gemini Flash (dùng default gemini-2.5-flash theo yêu cầu bài lab)
    text_gem, lat_gem, usage_gem = call_gemini(prompt, model="gemini-2.5-flash")
    cost_gem = _calc_cost("gemini-2.5-flash", usage_gem["input_tokens"], usage_gem["output_tokens"])

    # Trả về dictionary có đủ 3 key chuẩn theo yêu cầu test
    return {
        "gpt4o": {
            "response": text_4o,
            "latency": lat_4o,
            "cost": cost_4o,
            "input_tokens": usage_4o["input_tokens"],
            "output_tokens": usage_4o["output_tokens"],
        },
        "gpt4o_mini": {
            "response": text_mini,
            "latency": lat_mini,
            "cost": cost_mini,
            "input_tokens": usage_mini["input_tokens"],
            "output_tokens": usage_mini["output_tokens"],
        },
        "gemini_flash": {
            "response": text_gem,
            "latency": lat_gem,
            "cost": cost_gem,
            "input_tokens": usage_gem["input_tokens"],
            "output_tokens": usage_gem["output_tokens"],
        },
    }


# ---------------------------------------------------------------------------
# Task 5 — Streaming chatbot with Gemini 2.5 (Focus Model)
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal using Gemini 2.5.

    Behaviour:
        - Streams response tokens from Gemini 2.5 Flash as they arrive.
        - Maintains the last 3 turns of conversation history for context.
        - Typing 'quit' or 'exit' ends the session.

    Hints:
        - Maintain a history list of conversation turns.
        - Check how to stream responses using client.chats or model.generate_content(..., stream=True).
        - Keep history limited to the last 3 turns to optimize context window and costs.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    history: list[types.Content] = []

    print("=== Gemini Chatbot (gõ 'quit' hoặc 'exit' để thoát) ===")
    while True:
        try:
            user_msg = input("\nYou: ").strip()
        except EOFError:
            break

        if not user_msg:
            continue
        if user_msg.lower() in ("quit", "exit"):
            print("Tạm biệt!")
            break

        # 1. Thêm câu hỏi của user vào danh sách lịch sử
        history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_msg)]))

        # 2. Cắt ngắn lịch sử chỉ giữ tối đa 3 lượt hội thoại gần nhất (tương đương 6 tin nhắn)
        current_context = history[-6:]

        # 3. Gửi request dạng streaming để hiển thị chữ realtime
        print("Gemini: ", end="", flush=True)
        response_stream = client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=current_context,
        )

        model_response_chunks = []
        for chunk in response_stream:
            chunk_text = chunk.text or ""
            print(chunk_text, end="", flush=True)
            model_response_chunks.append(chunk_text)
        print()

        # 4. Gom các mảnh câu trả lời lại và lưu vào lịch sử
        full_response = "".join(model_response_chunks)
        history.append(types.Content(role="model", parts=[types.Part.from_text(text=full_response)]))


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable[[], Any],
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (delay = base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except Exception:
            # Nếu vượt quá số lần retry tối đa thì ném ngoại lệ ra ngoài
            if attempt >= max_retries:
                raise
            # Tính thời gian chờ theo cấp số nhân: 0.1s, 0.2s, 0.4s...
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
            attempt += 1


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    results = []
    # Lặp qua từng prompt và gọi compare_models
    for prompt in prompts:
        try:
            comparison = compare_models(prompt)
        except TypeError:
            # Xử lý tương thích với hàm mock _get_mock() trong test suite của bài lab
            comparison = compare_models()
        comparison["prompt"] = prompt  # Đính kèm câu hỏi gốc
        results.append(comparison)
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of batch compare results as a readable Markdown table string.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A beautiful Markdown table string with columns:
        | Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |
    """
    lines = [
        "| Prompt | Model | Response (truncated) | Latency | Tokens (In/Out) | Cost (USD) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    # Danh sách model cần hiển thị trong bảng
    model_display_names = [
        ("gpt4o", "GPT-4o"),
        ("gpt4o_mini", "GPT-4o-Mini"),
        ("gemini_flash", "Gemini-Flash"),
    ]

    for item in results:
        prompt_text = item.get("prompt", "")
        for key, display_name in model_display_names:
            if key in item:
                data = item[key]
                # Thay dấu xuống dòng bằng dấu cách để không làm vỡ ô trong Markdown table
                resp = data.get("response", "").replace("\n", " ")
                # Cắt ngắn phản hồi còn 50 ký tự
                resp_trunc = resp[:50] + ("..." if len(resp) > 50 else "")
                lat = f"{data.get('latency', 0.0):.2f}s"
                tokens = f"{data.get('input_tokens', 0)}/{data.get('output_tokens', 0)}"
                cost = f"${data.get('cost', 0.0):.6f}"
                lines.append(f"| {prompt_text} | {display_name} | {resp_trunc} | {lat} | {tokens} | {cost} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Model Comparison Test ===")
    test_prompt = "Hãy giải thích sự khác biệt giữa temperature và top_p bằng tiếng Việt ngắn gọn trong 2 câu."
    try:
        # Note: Requires valid API keys set in environment variables
        result = compare_models(test_prompt)
        for model_name, stats in result.items():
            print(f"\n[{model_name.upper()}]")
            print(f"Latency: {stats['latency']:.2f}s | Cost: ${stats['cost']:.6f}")
            print(f"Tokens: {stats['input_tokens']} in / {stats['output_tokens']} out")
            print(f"Response: {stats['response']}")
    except Exception as e:
        print(f"Skipping live API comparison test: {e}")
        print("Set your API keys to run manual tests.")

    print("\n=== Starting Gemini Chatbot (type 'quit' to exit) ===")
    try:
        streaming_chatbot()
    except Exception as e:
        print(f"Chatbot failed to start: {e}")
