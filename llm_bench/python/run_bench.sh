ROOT_DIR="/home/nlyaly/projects/lm-evaluation-harness/cache/phi-3-mini-4k-instruct"

declare -a EXP_DIRS=(
    'int4_sym_g128_r100_data_lora'
    'int4_sym_g128_r100_data_lora75'
    'int4_sym_g128_r100_data_lora50'
    'int4_sym_g128_r100_data_lora25'
    'int4_sym_g128_r100_data_lora_int8'
    'int4_sym_g128_r100_data_lora75_int8'
    'int4_sym_g128_r100_data_lora50_int8'
    'int4_sym_g128_r100_data_lora25_int8'
)

MAIN_CMD="numactl -N 0 --membind=0 python benchmark.py -mc 1 -ic 128 -d CPU -n 3 -p \"What is OpenVINO\" -lc dyn_quant_config.json"

for value in "${EXP_DIRS[@]}"
do
    MODEL_PATH=$ROOT_DIR/$value
    printf "\n\n$value\n"
    CMD_TO_RUN="$MAIN_CMD -m $MODEL_PATH"
    printf "\nLaunching $CMD_TO_RUN"
    eval $CMD_TO_RUN
done