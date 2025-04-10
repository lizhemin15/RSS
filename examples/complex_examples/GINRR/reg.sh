#!/bin/bash

# 默认任务数为4，如果指定了其他任务数则使用指定的值
NUM_TASKS=${1:-4}

# 初始化 GPU 列表
gpus=($(seq 0 $((NUM_TASKS - 1))))
gpu_count=${#gpus[@]}

# 定义要遍历的 reg_name, data_name, lap_k, mask_type 值
reg_names=('None' 'TV' 'INRR')
data_names=('baboon' 'cameraman' 'boats' 'man' 'pepers' 'walkbridge' 'woman' 'livingroom' 'house' 'jetplane' 'lake')
lap_k=('1' '2' '3' '4' '5')
mask_type=('random' 'patch' 'img')

# 计算总的任务数
total_tasks=0
for reg_name in "${reg_names[@]}"
do
    if [[ "$reg_name" == "INRR" || "$reg_name" == "Patch" ]]; then
        total_tasks=$((total_tasks + ${#data_names[@]} * ${#lap_k[@]} * ${#mask_type[@]}))
    else
        total_tasks=$((total_tasks + ${#data_names[@]} * 1 * ${#mask_type[@]}))
    fi
done
completed_tasks=0

# 创建任务队列
declare -A task_queue

# 定义一个函数来处理任务
run_task() {
    local reg_name=$1
    local data_name=$2
    local k=$3
    local mask=$4
    local gpu_id=$5

    echo "现在正在训练 reg_name 为 $reg_name, 数据集为 $data_name, lap_k 为 $k, mask_type 为 $mask 在GPU $gpu_id"
    python main.py --reg_name "$reg_name" --save_folder_name "reg" --data "$data_name" --lap_k "$k" --mask_type "$mask" --gpu_id "$gpu_id" --random_rate "0.5"
    echo "训练完毕: reg_name=$reg_name, 数据集=$data_name, lap_k=$k, mask_type=$mask 在GPU $gpu_id"

    completed_tasks=$((completed_tasks + 1))
    percent=$(( 100 * completed_tasks / total_tasks ))
    echo "当前进度: $percent%"
}

# 遍历所有组合并分配任务
assign_tasks() {
    for reg_name in "${reg_names[@]}"
    do
        for data_name in "${data_names[@]}"
        do
            if [[ "$reg_name" == "INRR" || "$reg_name" == "Patch" ]]; then
                for k in "${lap_k[@]}"
                do
                    for mask in "${mask_type[@]}"
                    do
                        # 运行任务
                        run_task "$reg_name" "$data_name" "$k" "$mask" "${gpus[0]}"
                    done
                done
            else
                k="1"
                for mask in "${mask_type[@]}"
                do
                    # 运行任务
                    run_task "$reg_name" "$data_name" "$k" "$mask" "${gpus[0]}"
                done
            fi
        done
    done
}

assign_tasks

# 定义消息内容
message_content="Reg训练完毕"

# 调用发送消息的脚本，并传递消息内容
./send.sh "$message_content"