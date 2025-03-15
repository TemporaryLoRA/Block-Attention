import json
import os
import ipdb


if __name__ == "__main__":
    for folder in os.listdir('data'):
        for file in os.listdir(os.path.join('data', folder)):
            if file in ['truthful_qa.json', 'ifeval.json', 'alpaca_eval.json']:
                path = os.path.join('data', folder, file)
                data = json.load(open(path))
                
                samples = {}
                for index, item in enumerate(data):
                    context = item['request'][0]['request']['context']
                    generation = item['request'][0]['request']['generated']
                    if item['label']:
                        gold = item['label']
                    else:
                        gold = {}
                    samples[index] = {
                        'origin_prompt': [{'role': "HUMAN", 'prompt': context}],
                        'prediction': generation,
                        'gold': gold
                    }

                if os.path.exists(os.path.join('output', folder)) is False:
                    os.makedirs(os.path.join('output', folder))

                if file == 'humaneval.json':
                    file = 'openai_humaneval.json'

                if file == 'ifeval.json':
                    file = 'IFEval.json'

                with open(os.path.join('output', folder, file.replace('.json', '_0.json')), 'w') as f:
                    json.dump(samples, f, ensure_ascii=False, indent=4)

            
            elif file.endswith('json') and file not in ['mmlu.json', 'bbh.json']:
                path = os.path.join('data', folder, file)
                data = json.load(open(path))
                
                samples = {}
                for index, item in enumerate(data):
                    context = item['request'][0]['request']['context']
                    generation = item['request'][0]['request']['generated']
                    if item['label']:
                        gold = item['label']
                    else:
                        try:
                            gold = item['doc']['test']
                        except:
                            ipdb.set_trace()
                    samples[index] = {
                        'origin_prompt': [{'role': "HUMAN", 'prompt': context}],
                        'prediction': generation.replace('<|end_of_text|>', '').strip('---'),
                        'gold': gold
                    }

                if os.path.exists(os.path.join('output', folder)) is False:
                    os.makedirs(os.path.join('output', folder))

                if file == 'humaneval.json':
                    file = 'openai_humaneval.json'

                with open(os.path.join('output', folder, file.replace('.json', '_0.json')), 'w') as f:
                    json.dump(samples, f, ensure_ascii=False, indent=4)
            elif file.endswith('json') and file in ['mmlu.json', 'bbh.json']:
                #ipdb.set_trace()
                path = os.path.join('data', folder, file)
                data = json.load(open(path))
                
                samples = {}
                for item in data:
                    splitted_data = item['native_id'].split()
                    assert len(splitted_data) == 2
                    subset_name, index = splitted_data
                    if subset_name not in samples:
                        samples[subset_name] = {}

                    context = item['request'][0]['request']['context']
                    generation = item['request'][0]['request']['generated']
                    if item['label']:
                        gold = item['label']
                    else:
                        try:
                            gold = item['doc']['test']
                        except:
                            ipdb.set_trace()
                    if file == 'bbh.json':
                        samples[subset_name][index] = {
                            'origin_prompt': [{'role': "HUMAN", 'prompt': context}],
                            'prediction': generation.replace('<|end_of_text|>', '').strip('---').strip('.'),
                            'gold': gold
                        }
                    else:
                        samples[subset_name][index] = {
                            'origin_prompt': [{'role': "HUMAN", 'prompt': context}],
                            'prediction': generation.replace('<|end_of_text|>', '').strip('---'),
                            'gold': gold
                        }

                if os.path.exists(os.path.join('output', folder)) is False:
                    os.makedirs(os.path.join('output', folder))

                for subset_name in samples:
                    if file == 'mmlu.json':
                        with open(os.path.join('output', folder, 'lukaemon_mmlu_' + subset_name + '_0.json'), 'w') as f:
                            json.dump(samples[subset_name], f, ensure_ascii=False, indent=4)
                    else:
                        with open(os.path.join('output', folder, 'bbh-' + subset_name + '_0.json'), 'w') as f:
                            json.dump(samples[subset_name], f, ensure_ascii=False, indent=4)

