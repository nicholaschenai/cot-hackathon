import tqdm
import sys
import os
from datetime import datetime
import logging
import json
import csv

from langchain.globals import set_debug, set_verbose

from cognitive_base.utils import lm_cache_init

from utils import process_single_solution, STOP_SEQUENCES
from eval_utils import compare_answers, load_math, extract_answer
from arg_parser import parse_arguments
from model_setup import create_agent
from prompts import get_system_prompt

from agent_tools import reset_persistent_globals
import traceback


def setup_logging(log_dir):
    log_file_path = os.path.join(log_dir, "master.log")
    
    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Create handlers
    file_handler = logging.FileHandler(log_file_path)
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Set level for handlers
    file_handler.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)
    
    # Create formatters and add them to handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return log_file_path

def save_args(args, log_dir):
    args_file_path = os.path.join(log_dir, "args.json")
    with open(args_file_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

def save_results(log_dir, results):
    results_file_path = os.path.join(log_dir, "comparison_results.csv")
    with open(results_file_path, 'w', newline='') as csvfile:
        fieldnames = ['i', 'level', 'problem_type', 'index', 'is_correct']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

def generate_replies(agent, problems, option, strategy, indices, stop_sequences=None):
    # TODO: need to restore the below thing for llama
    # prompts = [sys_prompt.replace(r"{problem}", problem) for problem in problems]
    # outputs = llm.batch(prompts)
    sys_prompt = get_system_prompt(option, strategy)
    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": problems[0]}]
    index = indices[0]
    traj_dir = os.path.join("log_files", timestamp, "trajs", str(index))
    os.makedirs(traj_dir, exist_ok=True)

    if strategy == "react":
        # WARNING: if we do parallel / batch, need to reset the persistent globals on a thread level
        reset_persistent_globals()
        inputs = {"messages": messages}
        # TODO: adjust logging so can capture intermediate steps, not done if error occurs eg max iter
        final_state = agent.invoke(inputs)
        generated_texts = [final_state["messages"][-1].content]

        for message in final_state["messages"]:
            message_str = message.pretty_repr() if not isinstance(message, tuple) else str(message)
            logging.debug(message_str)
            with open(os.path.join(traj_dir, "messages.log"), 'a') as f:
                f.write(message_str + "\n")
    else:
        inputs = messages
        output = agent.invoke(inputs)
        generated_texts = [output.content]

        for message in messages + [output]:
            message_str = message.pretty_repr() if not isinstance(message, dict) else message["content"]
            logging.debug(message_str)
            with open(os.path.join(traj_dir, "messages.log"), 'a') as f:
                f.write(message_str + "\n")

    solutions = []
    for generated_text in generated_texts:
        solution = process_single_solution(generated_text, stop_sequences if option == "local" else None)
        solutions.append(solution)
    return solutions

def main(agent, option, strategy, num_samples=None, seed=None, debug=False, batch_size=1):
    """
    Main evaluation loop.

    Args:
        num_samples (Optional[int]): Number of samples to evaluate. If None, evaluate all.
        seed (Optional[int]): Seed for reproducibility.
        batch_size (int): Batch size for processing samples.
    """
    def batch(ds, batch_size=batch_size):  # Use the batch_size parameter
        batch = ([], [], [], [], [])
        for i, e in enumerate(ds):
            batch[0].append(e[0])  # problem
            batch[1].append(e[1])  # solution
            batch[2].append(e[2])  # level
            batch[3].append(e[3])  # type
            batch[4].append(e[4])  # index
            if i % batch_size == 0 and i > 0:
                yield batch
                batch = ([], [], [], [], [])

    total, correct = 0, 0
    results = []
    for i, (problems, solutions, levels, types, indices) in enumerate(batch(tqdm.tqdm(load_math(num_samples=num_samples, seed=seed)))):
        try:
            model_answers = generate_replies(agent, problems, option, strategy, indices, stop_sequences=STOP_SEQUENCES if option == "local" else None)
        except Exception as e:
            logging.error(f"An error occurred: {e}", exc_info=True)
            model_answers = [""] * len(problems)

        for problem, solution, model_answer, level, problem_type, index in zip(problems, solutions, model_answers, levels, types, indices):
            # preprocess both answers
            y_true = extract_answer(solution)
            y_pred = extract_answer(model_answer)
            is_correct = compare_answers(y_true, y_pred)
            # Save y_true, y_pred, and is_correct to the respective trajectory directory
            traj_dir = os.path.join("log_files", timestamp, "trajs", str(index))
            with open(os.path.join(traj_dir, "answers.log"), 'a') as f:
                f.write(f"y_true: {y_true}, y_pred: {y_pred}, is_correct: {is_correct}\n")
            logging.debug(f"{y_true=}, {y_pred=}")
            correct += is_correct
            total += 1
            results.append({'i': i, 'level': level, 'problem_type': problem_type, 'index': index, 'is_correct': is_correct})
            logging.debug(f"Problem: {problem}\nModel Answer: {model_answer}\nTrue Answer: {solution}\nCorrect: {is_correct}\n")

        if (i + 1) % 10 == 0 or (i + 1) == num_samples:
            logging.info(f"Evaluated {i + 1}/{num_samples if num_samples else 'all'} samples. Current Accuracy: {correct/total:.2%}")
        logging.info(f"Accuracy: {correct}/{total} = {correct/total:.2%}")

    save_results(os.path.join("log_files", timestamp), results)
    

if __name__ == "__main__":
    args = parse_arguments()

    if not os.path.exists("log_files"):
        os.makedirs("log_files")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # log_file_path = os.path.join("log_files", f"{timestamp}.log")
    # log_file = open(log_file_path, "w")
    # sys.stdout = log_file
    run_log_dir = os.path.join("log_files", timestamp)
    os.makedirs(run_log_dir, exist_ok=True)
    os.makedirs(os.path.join(run_log_dir, "trajs"), exist_ok=True)

    setup_logging(run_log_dir)
    save_args(args, run_log_dir)

    # try:
    lm_cache_init('./lm_cache')
    set_verbose(True)
    set_debug(args.debug)
    agent = create_agent(args.option, args.strategy)
    main(agent, args.option, args.strategy, num_samples=args.num_samples, seed=args.seed, debug=args.debug, batch_size=args.batch_size)
    # finally:
    #     sys.stdout = sys.__stdout__
    #     log_file.close()
