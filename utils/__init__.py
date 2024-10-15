def process_single_solution(generated_text, stop_sequences=None):
    """
    Process a single generated text to extract the solution.

    Args:
        prompt (str): The original prompt.
        generated_text (str): The generated text from the model.
        stop_sequences (Optional[List[str]]): List of stop sequences to truncate the solution.

    Returns:
        str: The processed solution.
    """
    # Extract the solution part by removing the prompt
    solution = generated_text.strip()

    # Optional: Truncate at stop sequences if provided
    if stop_sequences:
        for stop_seq in stop_sequences:
            idx = solution.find(stop_seq)
            if idx != -1:
                solution = solution[:idx].strip()

    return solution

# NOTE: angle brackets break cursor composer application
STOP_SEQUENCES = ["<|eot_id|>"]