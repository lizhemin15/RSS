#!/bin/bash

# 定义要遍历的 reg_name, data_name, lap_k, noise_parameter 值
reg_names=('None' 'TV' 'INRR')
data_names=('baboon' 'cameraman' 'boats' 'man' 'pepers' 'walkbridge' 'woman' 'livingroom' 'house' 'jetplane' 'lake')
lap_k=('1' '2' '3' '4' '5')
noise_parameters=('5' '10' '15' '20' '25' '30' '35' '40')

# 设置 GPU ID
gpu=1

# 计算总的任务数
total_tasks=0
for reg_name in "${reg_names[@]}"
do
    if [[ "$reg_name" == "INRR" || "$reg_name" == "Patch" ]]; then
        total_tasks=$(( total_tasks + ${#data_names[@]} * ${#lap_k[@]} * ${#noise_parameters[@]} ))
    else
        total_tasks=$(( total_tasks + ${#data_names[@]} * ${#noise_parameters[@]} ))
    fi
done
completed_tasks=0

# 运行任务的函数
function run_task {
    reg_name=$1
    data_name=$2
    k=$3
    noise_parameter=$4

    # 增加进度百分比计算
    completed_tasks=$((completed_tasks + 1))
    percent=$(( 100 * completed_tasks / total_tasks ))
    
    echo "现在正在训练 reg_name 为 $reg_name, 数据集为 $data_name, lap_k 为 $k, noise_parameter 为 $noise_parameter, GPU 为 $gpu"
    echo "当前进度: $percent%"
    
    # 调用 main.py 并传递参数
    python main.py --reg_name "$reg_name" --save_folder_name "inv" --data "$data_name" --lap_k "$k" --gpu_id "$gpu" --random_rate "0." --noise_type "gaussian" --noise_parameter "$noise_parameter" --task "denoising"
    
    echo "训练完毕: reg_name=$reg_name, 数据集=$data_name, lap_k=$k, noise_parameter=$noise_parameter, GPU=$gpu"
}

# 使用 for 循环遍历这些值
for reg_name in "${reg_names[@]}"
do
    for data_name in "${data_names[@]}"
    do
        if [[ "$reg_name" == "INRR" || "$reg_name" == "Patch" ]]; then
            for k in "${lap_k[@]}"
            do
                for noise_parameter in "${noise_parameters[@]}"
                do
                    run_task "$reg_name" "$data_name" "$k" "$noise_parameter"
                done
            done
        else
            k=1
            for noise_parameter in "${noise_parameters[@]}"
            do
                run_task "$reg_name" "$data_name" "$k" "$noise_parameter"
            done
        fi
    done
done

# 定义消息内容
message_content="去噪问题训练完毕"

# 调用发送消息的脚本，并传递消息内容
./send.sh "$message_content"