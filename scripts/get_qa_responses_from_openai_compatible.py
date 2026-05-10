#!/usr/bin/env python3
"""Run QA examples through an OpenAI-compatible local or hosted model server."""
import argparse
import dataclasses
import json
import logging
import pathlib
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from common_io import xopen  # noqa: E402
from lost_in_the_middle.prompting import (  # noqa: E402
    Document,
    get_closedbook_qa_prompt,
    get_qa_prompt,
)
from openai_compatible_utils import (  # noqa: E402
    add_openai_compatible_args,
    clean_model_answer,
    complete_prompt,
    parse_extra_body,
)

logger = logging.getLogger(__name__)
random.seed(0)


def main(
    input_path,
    output_path,
    closedbook,
    prompt_mention_random_ordering,
    use_random_ordering,
    query_aware_contextualization,
    api_base,
    api_key,
    model_name,
    endpoint,
    system_prompt,
    temperature,
    top_p,
    max_new_tokens,
    stop,
    extra_body_json,
    request_timeout,
    max_retries,
    retry_sleep,
    num_workers,
    keep_thinking,
    max_examples,
):
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    extra_body = parse_extra_body(extra_body_json)

    examples = []
    prompts = []
    all_model_documents = []

    with xopen(input_path) as fin:
        for line_index, line in enumerate(tqdm(fin)):
            if max_examples is not None and line_index >= max_examples:
                break
            input_example = json.loads(line)
            prompt, documents = build_prompt(
                input_example=input_example,
                closedbook=closedbook,
                prompt_mention_random_ordering=prompt_mention_random_ordering,
                use_random_ordering=use_random_ordering,
                query_aware_contextualization=query_aware_contextualization,
            )
            prompts.append(prompt)
            examples.append(deepcopy(input_example))
            all_model_documents.append(documents)

    logger.info("Loaded %d prompts to process", len(prompts))

    generation_inputs = list(zip(examples, all_model_documents, prompts))

    def generate_output(generation_input):
        example, model_documents, prompt = generation_input
        raw_response = complete_prompt(
            prompt=prompt,
            api_base=api_base,
            api_key=api_key,
            model=model_name,
            endpoint=endpoint,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            system_prompt=system_prompt,
            extra_body=extra_body,
            stop=stop,
            timeout=request_timeout,
            max_retries=max_retries,
            retry_sleep=retry_sleep,
        )
        response = clean_model_answer(raw_response, strip_thinking=not keep_thinking)

        output_example = deepcopy(example)
        output_example["model_prompt"] = prompt
        output_example["model_documents"] = [dataclasses.asdict(document) for document in model_documents]
        output_example["model_answer"] = response
        output_example["model_raw_answer"] = raw_response
        output_example["model"] = model_name
        output_example["model_endpoint"] = endpoint
        output_example["model_temperature"] = temperature
        output_example["model_top_p"] = top_p
        output_example["model_prompt_mention_random_ordering"] = prompt_mention_random_ordering
        output_example["model_use_random_ordering"] = use_random_ordering
        output_example["model_query_aware_contextualization"] = query_aware_contextualization
        output_example["model_closedbook"] = closedbook
        return output_example

    with xopen(output_path, "w") as fout:
        if num_workers == 1:
            outputs = map(generate_output, generation_inputs)
            for output_example in tqdm(outputs, total=len(generation_inputs)):
                fout.write(json.dumps(output_example) + "\n")
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                outputs = executor.map(generate_output, generation_inputs)
                for output_example in tqdm(outputs, total=len(generation_inputs)):
                    fout.write(json.dumps(output_example) + "\n")


def build_prompt(
    input_example,
    closedbook,
    prompt_mention_random_ordering,
    use_random_ordering,
    query_aware_contextualization,
):
    question = input_example["question"]
    if closedbook:
        return get_closedbook_qa_prompt(question), []

    documents = [Document.from_dict(ctx) for ctx in deepcopy(input_example["ctxs"])]
    if not documents:
        raise ValueError(f"Did not find any documents for example: {input_example}")

    if use_random_ordering:
        (original_gold_index,) = [idx for idx, doc in enumerate(documents) if doc.isgold is True]
        original_gold_document = documents[original_gold_index]
        distractors = [doc for doc in documents if doc.isgold is False]
        random.shuffle(distractors)
        distractors.insert(original_gold_index, original_gold_document)
        documents = distractors

    prompt = get_qa_prompt(
        question,
        documents,
        mention_random_ordering=prompt_mention_random_ordering,
        query_aware_contextualization=query_aware_contextualization,
    )
    return prompt, documents


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", help="Path to data with questions and documents to use.", required=True)
    parser.add_argument("--output-path", help="Path to write output file of generated responses", required=True)
    parser.add_argument(
        "--closedbook", action="store_true", help="Run the model in closed-book mode (i.e., don't use documents)."
    )
    parser.add_argument(
        "--prompt-mention-random-ordering",
        action="store_true",
        help="Mention that search results are ordered randomly in the prompt",
    )
    parser.add_argument(
        "--use-random-ordering",
        action="store_true",
        help="Randomize the ordering of the distractors, rather than sorting by relevance.",
    )
    parser.add_argument(
        "--query-aware-contextualization",
        action="store_true",
        help="Place the question both before and after the documents.",
    )
    parser.add_argument("--max-examples", help="Maximum number of examples to process", type=int)
    add_openai_compatible_args(parser)
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))
    main(
        args.input_path,
        args.output_path,
        args.closedbook,
        args.prompt_mention_random_ordering,
        args.use_random_ordering,
        args.query_aware_contextualization,
        args.api_base,
        args.api_key,
        args.model,
        args.endpoint,
        args.system_prompt,
        args.temperature,
        args.top_p,
        args.max_new_tokens,
        args.stop,
        args.extra_body_json,
        args.request_timeout,
        args.max_retries,
        args.retry_sleep,
        args.num_workers,
        args.keep_thinking,
        args.max_examples,
    )
    logger.info("finished running %s", sys.argv[0])
