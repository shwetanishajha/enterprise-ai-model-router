
from src.adapters.openai_adapter import OpenAIAdapter


def main():
    adapter = OpenAIAdapter()

    result = adapter.generate_json(
        "Extract the revenue and EBITDA from this sentence: "
        "Revenue was £12.5 million and EBITDA was £2.1 million. "
        "Return JSON with numeric keys revenue_millions and ebitda_millions."
    )

    print("API request successful")
    print("Model:", result["model"])
    print("Output:", result["output"])
    print("Observed latency (ms):", result["latency_ms"])
    print("Input tokens:", result["input_tokens"])
    print("Output tokens:", result["output_tokens"])


if __name__ == "__main__":
    main()
