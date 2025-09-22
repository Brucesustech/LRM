#!/bin/bash


export CUDA_VISIBLE_DEVICES=0


config_dirs=(
    configs/GOODWebKB/university/concept
    # "configs/GOODWebKB/university/concept"
    # "configs/GOODTwitch/language/concept"
    # "configs/GOODCora/word/concept"
    # "configs/GOODCora/degree/concept"
)


num_runs=1


output_dir="experiment_results"
mkdir -p "$output_dir"
result_file="$output_dir/concept1_results.txt"


echo -e "\n\n\n" >> "$result_file"


for dir in "${config_dirs[@]}"; do
    yaml_files=$(find "$dir" -name "MARIO.yaml" -o -name "GRACE.yaml" -o -name "SWAV.yaml")
    # yaml_files=$(find "$dir" -name "GRACE.yaml")
    for config in $yaml_files; do
        echo "Running experiments for config: $config on GPU $CUDA_VISIBLE_DEVICES"
        
       
        base_name=$(dirname "$config")
        method_name=$(basename "$config" .yaml)

  
        dataset_path=$(echo "$base_name" | sed 's|^configs/||')
      
        dataset_path=$(echo "$dataset_path" | sed 's|/concept|-concept|')
        results_file="./storage/log/${dataset_path}/${method_name}-GCN_Encoder.log"

        echo "Expected results file: $results_file"

        python_config_path=$(echo "$config" | sed 's|^configs/||')
        echo "Using Python config path: $python_config_path"
       
        best_id_id_tests=()
        best_id_ood_tests=()
        best_ood_ood_tests=()

       
        for ((i=1; i<=num_runs; i++)); do
            echo "Run $i for $config on GPU $CUDA_VISIBLE_DEVICES"
            python unsupervised.py --config_path "$python_config_path" --ad_aug
            
           
            if [[ ! -f "$results_file" ]]; then
                echo "Error: $results_file not found. Skipping this run."
                continue
            fi

           
            results_line=$(grep "Results:" "$results_file" | tail -1)  
            echo "Extracted line: $results_line"

            if [[ $results_line =~ Results:\ ([0-9]+\.[0-9]+)\ ([0-9]+\.[0-9]+)\ ([0-9]+\.[0-9]+) ]]; then
                best_id_id_tests+=("${BASH_REMATCH[1]}")
                best_id_ood_tests+=("${BASH_REMATCH[2]}")
                best_ood_ood_tests+=("${BASH_REMATCH[3]}")
            else
                echo "Warning: No matching results found in $results_file"
            fi
        done

        if [[ ${#best_id_id_tests[@]} -eq 0 ]]; then
            mean_id_id="N/A"
            std_id_id="N/A"
        else
            mean_id_id=$(echo "${best_id_id_tests[@]}" | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i; print sum/NF}')
            std_id_id=$(echo "${best_id_id_tests[@]}" | awk '{sum=0; mean=0; n=NF; for(i=1;i<=n;i++) sum+=$i; mean=sum/n; sum=0; for(i=1;i<=n;i++) sum+=($i-mean)^2; print sqrt(sum/n)}')
        fi

        if [[ ${#best_id_ood_tests[@]} -eq 0 ]]; then
            mean_id_ood="N/A"
            std_id_ood="N/A"
        else
            mean_id_ood=$(echo "${best_id_ood_tests[@]}" | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i; print sum/NF}')
            std_id_ood=$(echo "${best_id_ood_tests[@]}" | awk '{sum=0; mean=0; n=NF; for(i=1;i<=n;i++) sum+=$i; mean=sum/n; sum=0; for(i=1;i<=n;i++) sum+=($i-mean)^2; print sqrt(sum/n)}')
        fi

        if [[ ${#best_ood_ood_tests[@]} -eq 0 ]]; then
            mean_ood_ood="N/A"
            std_ood_ood="N/A"
        else
            mean_ood_ood=$(echo "${best_ood_ood_tests[@]}" | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i; print sum/NF}')
            std_ood_ood=$(echo "${best_ood_ood_tests[@]}" | awk '{sum=0; mean=0; n=NF; for(i=1;i<=n;i++) sum+=$i; mean=sum/n; sum=0; for(i=1;i<=n;i++) sum+=($i-mean)^2; print sqrt(sum/n)}')
        fi

       
        full_config_path="$base_name/$(basename "$config")"
        
        
        echo -e "\n$full_config_path $mean_id_id ± $std_id_id, $mean_id_ood ± $std_id_ood, $mean_ood_ood ± $std_ood_ood" >> "$result_file"

        echo "Results saved to $result_file"
    done
done

echo "All experiments completed on GPU $CUDA_VISIBLE_DEVICES."