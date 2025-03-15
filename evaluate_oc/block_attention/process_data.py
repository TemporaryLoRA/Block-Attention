import json
from datasets import load_dataset
import csv
from copy import deepcopy
import os
import ipdb
from new_config import datasets
from mmlu_config import mmlu_all_sets
from bbh_config import bbh_free_form_sets, bbh_multiple_choice_sets


def get_answers(validated_answers):
    answers = []
    for answer_item in validated_answers:
        if answer_item['number']:
            answers.append(answer_item['number'])
        elif any(answer_item['date'][i] for i in ['day', 'month', 'year']):
            d = [answer_item['date'][i] for i in ['day', 'month', 'year']]
            answers.append(' '.join(d).strip())
        else:
            for span in answer_item['spans']:
                answers.append(span)
    answers = list(set(answers))
    return answers


def format(test_data, task_name):
    # format
    formatted_samples = []
    for id_, (sample, label, raw_sample, string) in test_data.items():
        new_sample = {
            'task_name': task_name,
            'label': label,
            'native_id': id_,
            "doc_id": id_,
            "doc": raw_sample,
            "request": [
                {
                    "request_type": "generate_util",
                    "request": {
                        "context": string,
                        "stop_sequences": ["Question", "</s>", "<|im_end|>"],
                        "generation_kwargs": {
                            "max_gen_toks": 2048,
                            "do_sample": False,
                            "temperature": 0.0
                        }
                    }
                }
            ]
        }
        formatted_samples.append(new_sample)
    return formatted_samples


def process_truthful_qa(data_path, config, task_name):
    tests, test_data = {}, {}
    tests = load_dataset('truthful_qa', 'generation')['validation']

    for index, sample in enumerate(tests):

        sample['reference'] = {
            'answers': {
                'best_answer': sample['best_answer'],
                'correct_answers': sample['correct_answers'],
                'incorrect_answers': sample['incorrect_answers']
            },
            'question': sample['question']
        }

        new_prompt = [{'role': 'HUMAN', 'prompt': sample['question']}]
        # convert prompt into string
        strings = []
        for index_utterance, utterance in enumerate(new_prompt):
            if utterance['role'] == 'HUMAN':
                strings.append('<|user|>\n' + utterance['prompt'] + '\n')
            elif utterance['role'] == 'BOT':
                if index_utterance == len(new_prompt) - 1:
                    strings.append('<|assistant|>\n' + utterance['prompt'] + '<|end_of_text|>')
                else:
                    strings.append('<|assistant|>\n' + utterance['prompt'] + '<|end_of_text|>\n')
            else:
                raise Exception("Unknown role:", utterance['role'])
        strings[-1] += '<|assistant|>\n'

        string = ''.join(strings)
        test_data[len(test_data)] = (new_prompt, "", sample, string)
    return format(test_data, task_name)

def process_ifeval(data_path, config, task_name):
    tests, test_data = {}, {}
    with open('ifeval_input_data.jsonl') as f:
        for line in f.readlines():
            tests[len(tests)] = json.loads(line)

    for index, sample in tests.items():
        new_prompt = [{'role': 'HUMAN', 'prompt': sample['prompt']}]
        # convert prompt into string
        strings = []
        for index_utterance, utterance in enumerate(new_prompt):
            if utterance['role'] == 'HUMAN':
                strings.append('<|user|>\n' + utterance['prompt'] + '\n')
            elif utterance['role'] == 'BOT':
                if index_utterance == len(new_prompt) - 1:
                    strings.append('<|assistant|>\n' + utterance['prompt'] + '<|end_of_text|>')
                else:
                    strings.append('<|assistant|>\n' + utterance['prompt'] + '<|end_of_text|>\n')
            else:
                raise Exception("Unknown role:", utterance['role'])
        strings[-1] += '<|assistant|>\n'



        string = ''.join(strings)
        test_data[len(test_data)] = (new_prompt, "", sample, string)
    return format(test_data, task_name)





def process_alpaca_eval(data_path, config, task_name):
    tests, test_data = {}, {}
    with open('alpaca_eval.json') as f:
        json_data = json.load(f)
        for problem in json_data:
            question = problem['question']
            capability = problem['capability']
            others = problem['others']
            tests[len(tests)] = {
                'question': question,
                'capability': capability,
                'others': others,
                'judge': {
                    'capability': capability,
                    'question': question
                }
            }

    for index, sample in tests.items():
        new_prompt = [{'role': 'HUMAN', 'prompt': sample['question']}]
        # convert prompt into string
        strings = []
        for index_utterance, utterance in enumerate(new_prompt):
            if utterance['role'] == 'HUMAN':
                strings.append('<|user|>\n' + utterance['prompt'] + '\n')
            elif utterance['role'] == 'BOT':
                if index_utterance == len(new_prompt) - 1:
                    strings.append('<|assistant|>\n' + utterance['prompt'] + '<|end_of_text|>')
                else:
                    strings.append('<|assistant|>\n' + utterance['prompt'] + '<|end_of_text|>\n')
            else:
                raise Exception("Unknown role:", utterance['role'])
        strings[-1] += '<|assistant|>\n'



        string = ''.join(strings)
        test_data[len(test_data)] = (new_prompt, "", sample, string)
    return format(test_data, task_name)


def process_humaneval(data_path, config, task_name):
    prompt = config['infer_cfg']['prompt_template']['template']['round']
    test_file = os.path.join(data_path, 'human-eval-v2-20210705.jsonl')
    test_data = {}
    for sample in open(test_file).readlines():
        sample = json.loads(sample)
        new_prompt = deepcopy(prompt)
        new_prompt[-1]['prompt'] = new_prompt[-1]['prompt'].format(prompt=sample['prompt'])
        # convert prompt into string
        strings = []
        for index_utterance, utterance in enumerate(new_prompt):
            if utterance['role'] == 'HUMAN':
                strings.append('<|user|>\n' + utterance['prompt'] + '\n')
            elif utterance['role'] == 'BOT':
                if index_utterance == len(new_prompt) - 1:
                    strings.append('<|assistant|>\n' + utterance['prompt'] + '<|end_of_text|>')
                else:
                    strings.append('<|assistant|>\n' + utterance['prompt'] + '<|end_of_text|>\n')
            else:
                raise Exception("Unknown role:", utterance['role'])
        strings[-1] += '<|assistant|>\n'



        string = ''.join(strings).rstrip('<|end_of_text|>')
        test_data[len(test_data)] = (new_prompt, "", sample, string)
    return format(test_data, task_name)


def process_gms8k(data_path, config, task_name):
    prompt = config['infer_cfg']['prompt_template']['template']['round']
    test_file = os.path.join(data_path, 'test.jsonl')
    test_data = {}
    for sample in open(test_file).readlines():
        sample = json.loads(sample)
        new_prompt = deepcopy(prompt)
        new_prompt[-1]['prompt'] = new_prompt[-1]['prompt'].format(question=sample['question'])
        # convert prompt into string
        strings = []
        for index_utterance, utterance in enumerate(new_prompt):
            if utterance['role'] == 'HUMAN':
                strings.append('<|user|>\n' + utterance['prompt'] + '\n')
            elif utterance['role'] == 'BOT':
                if index_utterance == len(new_prompt) - 1:
                    strings.append('<|assistant|>\n' + utterance['prompt'] + '<|end_of_text|>')
                else:
                    strings.append('<|assistant|>\n' + utterance['prompt'] + '<|end_of_text|>\n')
            else:
                raise Exception("Unknown role:", utterance['role'])
        strings[-1] += '<|assistant|>\n'


        string = ''.join(strings).rstrip('<|end_of_text|>')
        test_data[len(test_data)] = (new_prompt, sample['answer'], sample, string)
    return format(test_data, task_name)


def process_math(data_path, config, task_name):
    prompt = config['infer_cfg']['prompt_template']['template']['round']
    test_file = os.path.join(data_path, 'math.json')
    tests = json.load(open(test_file))
    test_data = {}
    for index, sample in tests.items():
        new_prompt = deepcopy(prompt)
        new_prompt[-1]['prompt'] = new_prompt[-1]['prompt'].format(problem=sample['problem'])
        # convert prompt into string
        strings = []
        for index_utterance, utterance in enumerate(new_prompt):
            if utterance['role'] == 'HUMAN':
                strings.append('<|user|>\n' + utterance['prompt'] + '\n')
            elif utterance['role'] == 'BOT':
                if index_utterance == len(new_prompt) - 1:
                    strings.append('<|assistant|>\n' + utterance['prompt'] + '<|end_of_text|>')
                else:
                    strings.append('<|assistant|>\n' + utterance['prompt'] + '<|end_of_text|>\n')
            else:
                raise Exception("Unknown role:", utterance['role'])
        strings[-1] += '<|assistant|>\n'




        string = ''.join(strings).rstrip('<|end_of_text|>')
        test_data[index] = (new_prompt, sample['solution'], sample, string)
    return format(test_data, task_name)


def process_drop(data_path, config, task_name):
    prompt = config['infer_cfg']['prompt_template']['template']['round']
    test_file = os.path.join(data_path, 'drop_dataset_dev.json')
    tests = json.load(open(test_file))
    test_data = {}
    index = 0
    for _, sample_ in tests.items():
        for qa_pair in sample_['qa_pairs']:
            ans = get_answers(qa_pair['validated_answers'])
            sample = {
                'prompt': sample_['passage'],
                'question': qa_pair['question'],
                'answers': ans
            }
            new_prompt = deepcopy(prompt)
            new_prompt[-1]['prompt'] = new_prompt[-1]['prompt'].format(prompt=sample['prompt'])
            # convert prompt into string
            strings = []
            for index_utterance, utterance in enumerate(new_prompt):
                if utterance['role'] == 'HUMAN':
                    strings.append('<|user|>\n' + utterance['prompt'] + '\n')
                elif utterance['role'] == 'BOT':
                    if index_utterance == len(new_prompt) - 1:
                        strings.append('<|assistant|>\n' + utterance['prompt'] + '<|end_of_text|>')
                    else:
                        strings.append('<|assistant|>\n' + utterance['prompt'] + '<|end_of_text|>\n')
                else:
                    raise Exception("Unknown role:", utterance['role'])
            strings[-1] += '<|assistant|>\n'


            string = ''.join(strings).rstrip('<|end_of_text|>')
            test_data[index] = (new_prompt, sample['answers'], sample, string)
            index += 1
    return format(test_data, task_name)


def process_mmlu(data_path, config, task_name):
    data_path = os.path.join(data_path, 'test')

    tests = {}
    for set_name in mmlu_all_sets:
        file_name = set_name + '_test.csv'
        file_name = os.path.join(data_path, file_name)
        with open(file_name, encoding='utf-8') as f:
            reader = csv.reader(f)
            for raw_index, row in enumerate(reader):
                assert len(row) == 6
                item = {
                    'input': row[0],
                    'A': row[1],
                    'B': row[2],
                    'C': row[3],
                    'D': row[4],
                    'target': row[5],
                    'set_name': set_name
                }
                raw_id = f'{set_name} {raw_index}'
                tests[raw_id] = item

    test_data = {}
    for index, sample in tests.items():
        set_name = sample['set_name']
        prompt = config[f'lukaemon_mmlu_{set_name}']['infer_cfg']['prompt_template']['template']['round']
        new_prompt = deepcopy(prompt)
        new_prompt[-1]['prompt'] = new_prompt[-1]['prompt'].format(input=sample['input'], A=sample['A'], B=sample['B'], C=sample['C'], D=sample['D'])
        # convert prompt into string
        strings = []
        for index_utterance, utterance in enumerate(new_prompt):
            if utterance['role'] == 'HUMAN':
                strings.append('<|user|>\n' + utterance['prompt'] + '\n')
            elif utterance['role'] == 'BOT':
                if index_utterance == len(new_prompt) - 1:
                    strings.append('<|assistant|>\n' + utterance['prompt'] + '<|end_of_text|>')
                else:
                    strings.append('<|assistant|>\n' + utterance['prompt'] + '<|end_of_text|>\n')
            else:
                raise Exception("Unknown role:", utterance['role'])
        strings[-1] += '<|assistant|>\n'


        string = ''.join(strings).rstrip('<|end_of_text|>')
        test_data[index] = (new_prompt, sample['target'], sample, string)
    return format(test_data, task_name)


def process_BBH(data_path, config, task_name):
    data_path = data_path.replace('bbh', 'BBH')
    hint_path = os.path.join(data_path, 'lib_prompt')
    data_path = os.path.join(data_path, 'data')

    prompts = {}

    tests = {}
    for set_name in bbh_multiple_choice_sets:
        with open(os.path.join(hint_path, f"{set_name}.txt")) as f:
            _hint = f.read()
        prompts[set_name] = [{
            'role': 'HUMAN',
            'prompt': f"Follow the given examples and answer the question.\n{_hint}\n\nQ:" + " {input}\nA: Let's think step by step."
        }]

        file_name = set_name + '.json'
        file_name = os.path.join(data_path, file_name)
        with open(file_name) as f:
            dd = json.load(f)['examples']
            for id_ in range(len(dd)):
                index = f'{set_name} {id_}'
                tests[index] = dd[id_]

    for set_name in bbh_free_form_sets:

        with open(os.path.join(hint_path, f"{set_name}.txt")) as f:
            _hint = f.read()
        prompts[set_name] = [{
            'role': 'HUMAN',
            'prompt': f"Follow the given examples and answer the question.\n{_hint}\n\nQ:" + " {input}\nA: Let's think step by step."
        }]

        file_name = set_name + '.json'
        file_name = os.path.join(data_path, file_name)
        with open(file_name) as f:
            dd = json.load(f)['examples']
            for id_ in range(len(dd)):
                index = f'{set_name} {id_}'
                tests[index] = dd[id_]

    test_data = {}
    for index, sample in tests.items():
        set_name = index.split()[0]
        prompt = prompts[set_name]
        new_prompt = deepcopy(prompt)
        new_prompt[-1]['prompt'] = new_prompt[-1]['prompt'].replace("{input}", sample['input'])
        # convert prompt into string
        strings = []
        for index_utterance, utterance in enumerate(new_prompt):
            if utterance['role'] == 'HUMAN':
                strings.append('<|user|>\n' + utterance['prompt'] + '\n')
            elif utterance['role'] == 'BOT':
                if index_utterance == len(new_prompt) - 1:
                    strings.append('<|assistant|>\n' + utterance['prompt'] + '<|end_of_text|>')
                else:
                    strings.append('<|assistant|>\n' + utterance['prompt'] + '<|end_of_text|>\n')
            else:
                raise Exception("Unknown role:", utterance['role'])
        strings[-1] += '<|assistant|>\n'




        string = ''.join(strings).rstrip('<|end_of_text|>')
        test_data[index] = (new_prompt, sample['target'], sample, string)
    return format(test_data, task_name)




if __name__ == "__main__":
    config_map = {}
    for item in datasets:
        config_map[item['abbr']] = item

    data_root_path = 'Please write down your OpenCompassData-Core folder path'
    data_output_path = 'data'
    mapping = {
        'gsm8k': process_gms8k,
        'humaneval': process_humaneval,
        'math': process_math,
        'drop': process_drop,
        'mmlu': process_mmlu,
        'bbh': process_BBH,
        'alpaca_eval': process_alpaca_eval,
        'ifeval': process_ifeval,
        'truthful_qa': process_truthful_qa
    }

    for dataset in mapping:
        if dataset == 'humaneval':
            cfg_dataset_name = 'openai_humaneval'
        else:
            cfg_dataset_name = dataset
        data_path = os.path.join(data_root_path, dataset)
        if dataset in ['mmlu', 'bbh', 'alpaca_eval', 'ifeval', 'truthful_qa']:
            processed_data = mapping[dataset](data_path, config_map, dataset)
        else:
            processed_data = mapping[dataset](data_path, config_map[cfg_dataset_name], dataset)
        output_path = os.path.join(data_output_path, dataset + '.json')
        with open(output_path, 'w') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=4)
