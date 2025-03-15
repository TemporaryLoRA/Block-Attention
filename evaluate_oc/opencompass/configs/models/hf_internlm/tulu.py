from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='tulu_oc',
        #path="/home/lt/NewPoNe/train/orm/internlm/add_deita/iter_6192_merge_hf/",
        max_out_len=1024,
        batch_size=1,
        model_kwargs=dict(device_map='auto'),
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='tulu_replicate_oc',
        #path="/home/lt/NewPoNe/train/orm/internlm/add_deita/iter_6192_merge_hf/",
        max_out_len=1024,
        batch_size=2,
        model_kwargs=dict(device_map='auto'),
        run_cfg=dict(num_gpus=1, num_procs=1),
    ),
]
