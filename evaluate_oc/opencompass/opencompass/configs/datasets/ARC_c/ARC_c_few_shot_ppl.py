from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import ARCDataset

ARC_c_reader_cfg = dict(
    input_columns=['question', 'textA', 'textB', 'textC', 'textD'],
    output_column='answerKey',
)

ARC_c_infer_cfg = dict(
    ice_template=dict(
        type=PromptTemplate,
        template={
            'A': dict(
                begin='</E>',
                round=[
                    dict(role='HUMAN', prompt='Question: {question}\nAnswer: '),
                    dict(role='BOT', prompt='{textA}'),
                ],
            ),
            'B': dict(
                begin='</E>',
                round=[
                    dict(role='HUMAN', prompt='Question: {question}\nAnswer: '),
                    dict(role='BOT', prompt='{textB}'),
                ],
            ),
            'C': dict(
                begin='</E>',
                round=[
                    dict(role='HUMAN', prompt='Question: {question}\nAnswer: '),
                    dict(role='BOT', prompt='{textC}'),
                ],
            ),
            'D': dict(
                begin='</E>',
                round=[
                    dict(role='HUMAN', prompt='Question: {question}\nAnswer: '),
                    dict(role='BOT', prompt='{textD}'),
                ],
            ),
        },
        ice_token='</E>',
    ),
    retriever=dict(type=FixKRetriever, fix_id_list=[0, 2, 4, 6, 8]),
    inferencer=dict(type=PPLInferencer),
)

ARC_c_eval_cfg = dict(evaluator=dict(type=AccEvaluator))

ARC_c_datasets = [
    dict(
        type=ARCDataset,
        abbr='ARC-c',
        path='opencompass/ai2_arc-dev',
        name='ARC-Challenge',
        reader_cfg=ARC_c_reader_cfg,
        infer_cfg=ARC_c_infer_cfg,
        eval_cfg=ARC_c_eval_cfg,
    )
]
