#!/bin/bash

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

# 计数器用于分配 GPU ID
gpu_id_counter=0

# 最大并行任务数
max_parallel_tasks=4

# 使用 for 循环遍历这些值
for reg_name in "${reg_names[@]}"
do
    for data_name in "${data_names[@]}"
    do
        if [[ "$reg_name" == "INRR" || "$reg_name" == "Patch" ]]; then
            for k in "${lap_k[@]}"
            do
                for mask in "${mask_type[@]}"
                do
                    # 增加进度百分比计算
                    completed_tasks=$((completed_tasks + 1))
                    percent=$(( 100 * completed_tasks / total_tasks ))
                    
                    echo "现在正在训练 reg_name 为 $reg_name, 数据集为 $data_name, lap_k 为 $k, mask_type 为 $mask"
                    echo "当前进度: $percent%"
                    
                    # 分配 GPU ID
                    gpu_id=$((gpu_id_counter % max_parallel_tasks))
                    gpu_id_counter=$((gpu_id_counter + 1))
                    
                    # 调用 main.py 并传递参数
                    python main.py --reg_name "$reg_name" --save_folder_name "rep" --data "$data_name" --lap_k "$k" --mask_type "$mask" --gpu_id "$gpu_id" --random_rate "0.5" --dmf_depth "3" &
                    
                    # 限制并行任务数
                    if (( gpu_id_counter % max_parallel_tasks == 0 )); then
                        wait
                    fi
                    
                    echo "训练完毕: reg_name=$reg_name, 数据集=$data_name, lap_k=$k, mask_type=$mask"
                done
            done
        else
            k="1"
            for mask in "${mask_type[@]}"
            do
                # 增加进度百分比计算
                completed_tasks=$((completed_tasks + 1))
                percent=$(( 100 * completed_tasks / total_tasks ))
                
                echo "现在正在训练 reg_name 为 $reg_name, 数据集为 $data_name, lap_k 为 $k, mask_type 为 $mask"
                echo "当前进度: $percent%"
                
                # 分配 GPU ID
                gpu_id=$((gpu_id_counter % max_parallel_tasks))
                gpu_id_counter=$((gpu_id_counter + 1))
                
                # 调用 main.py 并传递参数
                python main.py --reg_name "$reg_name" --save_folder_name "rep" --data "$data_name" --lap_k "$k" --mask_type "$mask" --gpu_id "$gpu_id" --random_rate "0.5" --dmf_depth "3" &
                
                # 限制并行任务数
                if (( gpu_id_counter % max_parallel_tasks == 0 )); then
                    wait
                fi
                
                echo "训练完毕: reg_name=$reg_name, 数据集=$data_name, lap_k=$k, mask_type=$mask"
            done
        fi
    done
done

# 等待所有后台任务完成
wait

# 定义消息内容
message_content="Representation 训练完毕"

# 调用发送消息的脚本，并传递消息内容
./send.sh "$message_content"