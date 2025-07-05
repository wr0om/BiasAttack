import os
import gc
import json
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
# from livelossplot import PlotLosses
import matplotlib.pyplot as plt
import time
from collections import Counter

# Helper function imports (unchanged)
from llm_attacks.minimal_gcg.opt_utils import (
    token_gradients, sample_control, get_logits, target_loss,
    load_model_and_tokenizer as external_load_model_and_tokenizer,
    get_filtered_cands
)
from llm_attacks.minimal_gcg.string_utils import (
    SuffixManager, load_conversation_template
)
from llm_attacks import get_nonascii_toks
# from nanogcg import GCGConfig
from bias_utils import (
    test_prefixes,
    model_bias_answers_prefixes,
    model_path_dicts,
    get_developer,
    Args,
    load_and_save_final_suffixes,
    check_for_attack_success,
    find_answer,
    log_init,
    load_model_and_tokenizer,
    load_conversation_template as bias_load_conversation_template,
    generate
)

# Manually empty the cache before loading (unchanged)
gc.collect()
torch.cuda.empty_cache()

# Add to the prefixes: colon variants, space variants, etc. (unchanged)
colon_prefixes = [f"{prefix}:" for prefix in model_bias_answers_prefixes]
space_prefixes = [f"{prefix} " for prefix in model_bias_answers_prefixes]
s_prefixes = [f"{prefix}<\\s>" for prefix in model_bias_answers_prefixes]
model_bias_answers_prefixes += colon_prefixes + space_prefixes

# Set random seeds for reproducibility (unchanged)
seed = 20
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Sub-function: Attack logic for an individual prompt
def attack_individual(
    row,
    model,
    tokenizer,
    conv_template,
    model_name,
    test_prefixes,
    model_bias_answers_prefixes,
    device,
    num_steps,
    batch_size,
    topk,
    not_allowed_tokens,
    suffix,
    args,
    number_samples=10  # Set to 10 samples per pattern
):
    """
    Runs the bias attack on a single row (prompt) from the dataset.
    Collects exactly number_samples (10) samples for this prompt.
    """
    user_prompt = row['goal']
    target = row['target']
    log = log_init()

    # Initialize tracking variables
    success = False
    answer = None
    final_suffix = ""
    adv_suffix = suffix
    # Keep a tokenized copy so we don't lose tokens when decoding/encoding again
    adv_suffix_tokens = tokenizer(adv_suffix, add_special_tokens=False).input_ids
    answers_list = []
    total_no_grad_time = 0

    members_part = user_prompt.split("'")[1].split("'")[0]
    members = [member.strip() for member in members_part.split(",")]
    
    suffix_manager = SuffixManager(
        tokenizer=tokenizer,
        conv_template=conv_template,
        instruction=user_prompt,
        target=target,
        adv_string=adv_suffix
    )
    input_ids = suffix_manager.get_input_ids(adv_string=adv_suffix).to(device)
    # Split the prompt tokens around the control slice so we can rebuild it
    pre_control_tokens = input_ids[:suffix_manager._control_slice.start]
    post_control_tokens = input_ids[suffix_manager._control_slice.stop:]
    print(f"Starting with suffix: {adv_suffix}")
    if num_steps == 0:
        raise ValueError("num_steps must be nonzero")

    answers_histogram = {member: 0 for member in members}

    for i in range(num_steps):
        # Reconstruct the suffix string from the current token list for logging
        adv_suffix = tokenizer.decode(adv_suffix_tokens)
        print(f"(4) step {i}: length of adv_suffix_tokens: {len(adv_suffix_tokens)}")

        input_ids = torch.cat([
            pre_control_tokens,
            torch.tensor(adv_suffix_tokens, dtype=pre_control_tokens.dtype),
            post_control_tokens
        ]).to(device)

        coordinate_grad = token_gradients(
            model,
            input_ids,
            suffix_manager._control_slice,
            suffix_manager._target_slice,
            suffix_manager._loss_slice
        )

        no_grad_start_time = time.time()
        with torch.no_grad():
            # (2) TODO: is it the control slice?
            # Finding: control slice upper bound also goes down by 1 at each step.
            # print(f"Control slice: {suffix_manager._control_slice}")

            adv_suffix_tokens_tensor = input_ids[suffix_manager._control_slice].to(device)


            # (3) TODO: why does adv_suffix_tokens change?
            # Finding: sample_control is NOT the problem.
            len_suffix = len(adv_suffix_tokens_tensor)
            print(f"adv_suffix_tokens length: {len_suffix}")

            new_adv_suffix_toks = sample_control(
                adv_suffix_tokens_tensor,
                coordinate_grad,
                batch_size,
                topk=topk,
                temp=1,
                not_allowed_tokens=not_allowed_tokens
            )

            # (1) TODO: why suffix sizes change? 
            # Finding: suffix goes down by 1 at each step!! Find this bug!
            # for i in range(len(new_adv_suffix_toks)):
            #     len_suffix = len(new_adv_suffix_toks[i])
            #     print(f"new_adv_suffix_toks {i} length: {len_suffix}")

            try:
                new_adv_suffix = get_filtered_cands(
                    tokenizer,
                    new_adv_suffix_toks,
                    filter_cand=True,
                    curr_control=adv_suffix
                )
            except Exception as e:
                print(f"Error in get_filtered_cands: {e}")
                i -= 1
                continue

            logits, ids = get_logits(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                control_slice=suffix_manager._control_slice,
                test_controls=new_adv_suffix,
                return_ids=True,
                batch_size=batch_size
            )

            losses = target_loss(logits, ids, suffix_manager._target_slice)

            best_new_adv_suffix_id = losses.argmin()
            best_new_adv_suffix = new_adv_suffix[best_new_adv_suffix_id]
            current_loss = losses[best_new_adv_suffix_id]

            # Keep tokens from the chosen candidate for the next iteration
            adv_suffix_tokens = new_adv_suffix_toks[best_new_adv_suffix_id].tolist()
            adv_suffix = best_new_adv_suffix

            is_success, gen_str = check_for_attack_success(
                model, tokenizer, input_ids, suffix_manager._assistant_role_slice, test_prefixes
            )
            print_progress(is_success, best_new_adv_suffix, gen_str, answer, i,
                           len(answers_list), target, user_prompt, current_loss)

        no_grad_end_time = time.time()
        total_no_grad_time += (no_grad_end_time - no_grad_start_time)

        answer = find_answer(gen_str, user_prompt, model_bias_answers_prefixes, members)
        if answer is not None:
            print('####################\n')
            print(f"Answer found: {answer}")
            print('####################\n')

        if answer is not None:
            success = True
            final_suffix = adv_suffix
            for member in members:
                if member.lower() in answer.lower():
                    answers_histogram[member] += 1
                    answers_list.append(member)

        log["loss"].append(current_loss.item())
        log["suffix"].append(adv_suffix)
        log["time"].append(time.time())
        log["respond"].append(answer)
        log["success"].append(is_success)
        del coordinate_grad
        gc.collect()
        torch.cuda.empty_cache()

        if len(answers_list) >= number_samples:  # Stop after 10 samples for this pattern
            print(f"Collected {number_samples} samples for prompt, stopping...")
            break

    result = {
        "user_prompt": user_prompt,
        "target": target,
        "success": success,
        "final_suffix": final_suffix,
        "answer": answer,
        "answers_histogram": answers_histogram,
        "answers_count": len(answers_list),
        "attack_time": total_no_grad_time,
        "log": log
    }
    return result

# Sub-function: Process a group of files
# --------------------------------------------------------------------------- #
#  Updated run_on_group                                                       #
# --------------------------------------------------------------------------- #
def run_on_group(
    group,
    dataset_dir,
    results_dir,
    model,
    tokenizer,
    conv_template,
    model_name,
    test_prefixes,
    model_bias_answers_prefixes,
    device,
    num_steps,
    batch_size,
    topk,
    not_allowed_tokens,
    suffix,
    args,
    number_samples=10  # 10 samples per pattern
):
    """
    Processes all CSV files in a group directory, running the attack on each row.
    Ensures 10 samples per question pattern, totaling 100 samples per subject.
    """
    group_dir = os.path.join(dataset_dir, group)
    results = []
    suffix_init = suffix
    for root, _, files in os.walk(group_dir):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}")
                data = pd.read_csv(file_path)

                save_path = os.path.join(
                    results_dir,
                    os.path.relpath(file_path, dataset_dir).replace(".csv", ".json")
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                # Load existing results
                existing_results = []
                if os.path.exists(save_path):
                    with open(save_path, "r") as f:
                        existing_results = json.load(f)
                    print(f"Loaded {len(existing_results)} existing results from {save_path}")

                results = existing_results

                # Check number of unique prompts
                unique_prompts = data['goal'].nunique()
                print(f"File {file_path} has {len(data)} rows with {unique_prompts} unique prompts")
                if unique_prompts != 10:
                    print(f"Warning: Expected 10 unique prompts, found {unique_prompts}")

                for idx, row in tqdm(data.iterrows(), total=len(data), desc=f"Processing {file_path}"):
                    skip = False
                    target = row['target']
                    user_prompt = row['goal']

                    # Check if this specific prompt has enough samples
                    for result in results:
                        if result['user_prompt'] == user_prompt and result.get("answers_count", 0) >= number_samples:
                            print(f"Skipping prompt '{user_prompt}' as it has {result['answers_count']} samples")
                            skip = True
                            break
                        elif result['user_prompt'] == user_prompt:
                            # Use the last suffix if partially processed
                            suffix = result["final_suffix"]

                    if skip:
                        continue

                    print(f"Processing group {group}, file {file}")
                    print(f"Target: {target}")
                    print(f"Prompt: {user_prompt}")
                    result = attack_individual(
                        row, model, tokenizer, conv_template, model_name, test_prefixes,
                        model_bias_answers_prefixes, device, num_steps, batch_size, topk,
                        not_allowed_tokens, suffix, args, number_samples=number_samples
                    )
                    results.append(result)
                    save_results(results, save_path)

    return results

# Sub-function: Save results to JSON
def save_results(results, save_path):
    """
    Saves the attack results to a JSON file.
    """
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)
    torch.cuda.empty_cache()
    gc.collect()

# Sub-function: Print progress information
def print_progress(is_success, suffix, gen_str, answer, step, size_answers_list, target, user_prompt, current_loss):
    """
    Prints progress information during the attack.
    """
    print(f"Target: {target}")
    print(f"Prompt: {user_prompt}")
    print(f"Current loss: {current_loss}")
    print(f"Answer list length: {size_answers_list}")
    print(f"\nPassed:{is_success}\nCurrent Suffix:{suffix}", end='\r')
    print(f"\nGen Str: {gen_str}\n", end='\r')
    print(f"\nAnswer: {answer}\n", end='\r')
    print(f"\nStep: {step}\n", end='\r')


def run_attack_on_groups(
    group_names,
    dataset_dir,
    results_dir,
    model,
    tokenizer,
    num_steps=100,
    batch_size=16,
    model_name="llama2",
    test_prefixes=test_prefixes,
    args=None,
    temp=2,
    model_bias_answers_prefixes=model_bias_answers_prefixes,
    topk=128,
    number_samples=10  # 10 samples per pattern
):
    device = "cuda:" + str(args.device)
    print(f"Running attack on groups: {group_names}")
    results_dir = args.results_dir
    topk = args.topk
    batch_size = args.batch_size
    num_steps = args.num_steps

    not_allowed_tokens = get_nonascii_toks(tokenizer)
    adv_suffix = ""
    conv_template = load_conversation_template(model_name)
    crit = nn.CrossEntropyLoss(reduction='mean')
    suffix = args.suffix
    if suffix is None:
        suffix = 20 * "hi "

    for group in group_names:
        run_on_group(
            group, dataset_dir, results_dir, model, tokenizer, conv_template,
            model_name, test_prefixes, model_bias_answers_prefixes, device,
            num_steps, batch_size, topk, not_allowed_tokens, suffix, args,
            number_samples=number_samples
        )

def run_bias(
    model,
    tokenizer,
    test_prefixes=test_prefixes,
    model_bias_answers_prefixes=model_bias_answers_prefixes,
    args=None
):
    model_name = args.model_name
    number_samples = 10  # 10 samples per pattern
    run_attack_on_groups(
        group_names=args.group_names,
        dataset_dir=args.data_dir,
        results_dir=args.results_dir,
        model=model,
        tokenizer=tokenizer,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        model_name=args.model_name,
        test_prefixes=test_prefixes,
        args=args,
        number_samples=number_samples
    )



# Main entry point (unchanged)
if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()

    args = Args()  # This will now automatically parse command line arguments

    chosen_model_path = model_path_dicts.get(args.model_name, args.model_name)
    print(f"Using model path: {chosen_model_path}")

    if args.group_names is None:
        args.group_names = ['abilities', 'ages', 'body types', 'characteristics', 'cultures', 'genders', 'nationalities', 'nonces', 'political ideologies', 'races', 'religions', 'sexual orientations', 'socioeconomic classes', 'Politicians', 'Tech Celebrities']

    results_dir = args.results_dir
    print(f"Results will be saved to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)
    device_str = "cuda:" + str(args.device)

    save_directory = "./models"
    model, tokenizer = load_model_and_tokenizer(chosen_model_path, save_directory, device=device_str)

    print(f"Model Name: {args.model_name}")
    print(f"Group Names: {args.group_names}")

    run_bias(model, tokenizer, test_prefixes=test_prefixes, model_bias_answers_prefixes=model_bias_answers_prefixes, args=args)