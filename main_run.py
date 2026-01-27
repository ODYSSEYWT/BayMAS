import argparse
import os


parser=argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="HumanEval", choices=["HumanEval", "MBPP", "APPS", "MMLU"])
parser.add_argument("--strategy", type=str, default="MapCoder", choices=["Direct", "DirectTest", "CoT", "SelfPlanning", "SelfPlanningTest", "SelfPlanningTestNonRel", "SelfPlanningTestNegRel", "Analogical", "MapCoder", "SelfPlanningTestSnowballing", "SelfPlanningTestNonRelSnowballing", "SelfPlanningTestNegRelSnowballing"])
parser.add_argument("--models", type=str, default="Qwen/Qwen3-4B-Instruct-2507", choices=["Qwen/Qwen3-Coder-30B-A3B-Instruct", "Qwen/Qwen2.5-Coder-14B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen/Qwen2.5-Coder-3B-Instruct", "Qwen/Qwen2.5-Coder-1.5B-Instruct", "Qwen/Qwen2.5-Coder-0.5B-Instruct", "Qwen/Qwen2.5-Coder-32B-Instruct", "Qwen/Qwen3-30B-A3B-Instruct-2507", "Qwen/Qwen3-4B-Instruct-2507", "Qwen/Qwen3-4B", "Qwen/Qwen3-30B-A3B", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "openai/gpt-oss-120b", "Qwen/Qwen3-14B", "Qwen/Qwen3-8B", "Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "deepseek-ai/deepseek-coder-33b-instruct", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "meta-llama/Llama-3.1-70B-Instruct", "meta-llama/Llama-3.1-70B-Instruct"])
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--num_return_sequences", type=int, default=1)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--agent_num', type=str, default='2')
parser.add_argument("--type", type=str, required=True, choices=["gen", "pe", "se", "sup", "lars", "bs", "sar", "saup", "pe_swarm", "bs_swarm", "sar_swarm", "sup_swarm", "se_swarm", "saup_swarm", "bo_swarm", "bog_swarm", "lars_swarm", "bo_v1_grad_swarm", "bog_v1_grad_swarm", "pe_mmlu", "bs_mmlu", "sar_mmlu", "sup_mmlu", "se_mmlu", "saup_mmlu", "bo_mmlu", "lars_mmlu", "bog_mmlu", "bog_v1_grad_v2", "bog_v1_grad_v2_swarm", "bog_v1_grad_v2_mmlu", "bog_v1_grad_v2_ablation", "bog_v1_grad_v2_ablation_swarm", "bog_v1_grad_v2_ablation_mmlu", "bog_v1_grad_v2_time", "bog_v1_grad_v2_time_swarm", "bog_v1_grad_v2_time_mmlu", "pe_time", "pe_time_swarm", "pe_time_mmlu", "bs_time", "bs_time_swarm", "bs_time_mmlu", "sup_time", "sup_time_swarm", "sup_time_mmlu", "sar_time", "sar_time_swarm", "sar_time_mmlu", "se_time", "se_time_swarm", "se_time_mmlu", "saup_time", "saup_time_swarm", "saup_time_mmlu", "lars_time", "lars_time_swarm", "lars_time_mmlu", "bog_v1_grad_v2_scalability", "bog_v1_grad_v2_scalability_swarm", "bog_v1_grad_v2_scalability_mmlu"], help="type of method to run")
parser.add_argument("--mmlu_type", type=str, choices=["FullConnectedSwarm", "RandomConnectedSwarm"])
parser.add_argument("--random_split", type=int, default=42)
parser.add_argument("--sample_ratio", type=float, default=1.0)

args=parser.parse_args()

from multiprocessing import Process

def task(dataset, model):
    if args.type == "gen":
        log_path = f"logs/{dataset}/Gen/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}-{str(args.num_return_sequences)}.log"
        command=f"python src/main.py --model {model} --dataset {dataset} --strategy {args.strategy} --temperature {str(args.temperature)} --device {args.gpu} --num_return_sequences {str(args.num_return_sequences)}"
        # command=f"nohup python src/main.py --model {model} --dataset {dataset} --strategy {args.strategy} --temperature {str(args.temperature)} --device {args.gpu} --num_return_sequences {str(args.num_return_sequences)} > {log_path} 2>&1 &"
    elif args.type == "pe":
        log_path = f"logs/{dataset}/PE/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/PE.py --model {model} --dataset {dataset} --strategy {args.strategy} --temperature {str(args.temperature)} --device {args.gpu} --random_split {str(args.random_split)}"
    elif args.type == "pe_swarm":
        log_path = f"logs/{dataset}/PE-swarm/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/PE_swarm.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --random_split {str(args.random_split)}"
    elif args.type == "pe_mmlu":
        log_path = f"logs/{dataset}/PE-MMLU/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/PE_MMLU.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --type {args.mmlu_type} --random_split {str(args.random_split)}"
    elif args.type == "pe_time":
        log_path = f"logs/{dataset}/PE/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/PE_time.py --model {model} --dataset {dataset} --strategy {args.strategy} --temperature {str(args.temperature)} --device {args.gpu} --random_split {str(args.random_split)}"
    elif args.type == "pe_time_swarm":
        log_path = f"logs/{dataset}/PE-swarm/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/PE_swarm_time.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --random_split {str(args.random_split)}"
    elif args.type == "pe_time_mmlu":
        log_path = f"logs/{dataset}/PE-MMLU/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/PE_MMLU_time.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --type {args.mmlu_type} --random_split {str(args.random_split)}"
    
    elif args.type == "bog_v1_grad_v2":
        log_path = f"logs/{dataset}/BOG_v1_grad_v2/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/BOG_v1_grad_tmp_v2.py --model {model} --dataset {dataset} --strategy {args.strategy} --temperature {str(args.temperature)} --device {args.gpu} --random_split {str(args.random_split)}"
    elif args.type == "bog_v1_grad_v2_swarm":
        log_path = f"logs/{dataset}/BOG_v1_grad_v2/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/BOG_v1_grad_tmp_swarm_v2.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --random_split {str(args.random_split)}"
    elif args.type == "bog_v1_grad_v2_mmlu":
        log_path = f"logs/{dataset}/BOG_v1_grad_v2/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/BOG_v1_grad_tmp_MMLU_v2.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --type {args.mmlu_type} --random_split {str(args.random_split)}"
    elif args.type == "bog_v1_grad_v2_ablation":
        log_path = f"logs/{dataset}/BOG_v1_grad_v2/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/BOG_v1_grad_tmp_v2_ablation.py --model {model} --dataset {dataset} --strategy {args.strategy} --temperature {str(args.temperature)} --device {args.gpu} --random_split {str(args.random_split)}"
    elif args.type == "bog_v1_grad_v2_ablation_swarm":
        log_path = f"logs/{dataset}/BOG_v1_grad_v2/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/BOG_v1_grad_tmp_swarm_v2_ablation.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --random_split {str(args.random_split)}"
    elif args.type == "bog_v1_grad_v2_ablation_mmlu":
        log_path = f"logs/{dataset}/BOG_v1_grad_v2/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/BOG_v1_grad_tmp_MMLU_v2_ablation.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --type {args.mmlu_type} --random_split {str(args.random_split)}"
    elif args.type == "bog_v1_grad_v2_time":
        log_path = f"logs/{dataset}/BOG_v1_grad_v2/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/BOG_v1_grad_tmp_v2_time.py --model {model} --dataset {dataset} --strategy {args.strategy} --temperature {str(args.temperature)} --device {args.gpu} --random_split {str(args.random_split)}"
    elif args.type == "bog_v1_grad_v2_time_swarm":
        log_path = f"logs/{dataset}/BOG_v1_grad_v2/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/BOG_v1_grad_tmp_swarm_v2_time.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --random_split {str(args.random_split)}"
    elif args.type == "bog_v1_grad_v2_time_mmlu":
        log_path = f"logs/{dataset}/BOG_v1_grad_v2/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/BOG_v1_grad_tmp_MMLU_v2_time.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --type {args.mmlu_type} --random_split {str(args.random_split)}"
    elif args.type == "bog_v1_grad_v2_scalability":
        log_path = f"logs/{dataset}/BOG_v1_grad_v2/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/BOG_v1_grad_tmp_v2_scalability.py --model {model} --dataset {dataset} --strategy {args.strategy} --temperature {str(args.temperature)} --device {args.gpu} --random_split {str(args.random_split)} --sample_ratio {str(args.sample_ratio)}"
    elif args.type == "bog_v1_grad_v2_scalability_swarm":
        log_path = f"logs/{dataset}/BOG_v1_grad_v2/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/BOG_v1_grad_tmp_swarm_v2_scalability.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --random_split {str(args.random_split)} --sample_ratio {str(args.sample_ratio)}"
    elif args.type == "bog_v1_grad_v2_scalability_mmlu":
        log_path = f"logs/{dataset}/BOG_v1_grad_v2/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/BOG_v1_grad_tmp_MMLU_v2_scalability.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --type {args.mmlu_type} --random_split {str(args.random_split)} --sample_ratio {str(args.sample_ratio)}"
    
    elif args.type == "bs":
        log_path = f"logs/{dataset}/BS/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/BS.py --model {model} --dataset {dataset} --strategy {args.strategy} --temperature {str(args.temperature)} --device {args.gpu} --random_split {str(args.random_split)}"
    elif args.type == "bs_swarm":
        log_path = f"logs/{dataset}/BS-swarm/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/BS_swarm.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --random_split {str(args.random_split)}"
    elif args.type == "bs_mmlu":
        log_path = f"logs/{dataset}/BS-MMLU/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/BS_MMLU.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --type {args.mmlu_type} --random_split {str(args.random_split)}"
    elif args.type == "bs_time":
        log_path = f"logs/{dataset}/BOG_v1_grad_v2/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/BS_time.py --model {model} --dataset {dataset} --strategy {args.strategy} --temperature {str(args.temperature)} --device {args.gpu} --random_split {str(args.random_split)}"
    elif args.type == "bs_time_swarm":
        log_path = f"logs/{dataset}/BOG_v1_grad_v2/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/BS_swarm_time.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --random_split {str(args.random_split)}"
    elif args.type == "bs_time_mmlu":
        log_path = f"logs/{dataset}/BOG_v1_grad_v2/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/BS_MMLU_time.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --type {args.mmlu_type} --random_split {str(args.random_split)}"

    elif args.type == "sup":
        log_path = f"logs/{dataset}/SUP/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/Sup.py --model {model} --dataset {dataset} --strategy {args.strategy} --temperature {str(args.temperature)} --random_split {str(args.random_split)}"
    elif args.type == "sup_swarm":
        log_path = f"logs/{dataset}/SUP-swarm/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/Sup_swarm.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --random_split {str(args.random_split)}"
    elif args.type == "sup_mmlu":
        log_path = f"logs/{dataset}/SUP-MMLU/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/Sup_MMLU.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --type {args.mmlu_type} --random_split {str(args.random_split)}"
    elif args.type == "sup_time":
        log_path = f"logs/{dataset}/SUP/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/Sup_time.py --model {model} --dataset {dataset} --strategy {args.strategy} --temperature {str(args.temperature)} --random_split {str(args.random_split)}"
    elif args.type == "sup_time_swarm":
        log_path = f"logs/{dataset}/SUP-swarm/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/Sup_swarm_time.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --random_split {str(args.random_split)}"
    elif args.type == "sup_time_mmlu":
        log_path = f"logs/{dataset}/SUP-MMLU/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/Sup_MMLU_time.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --type {args.mmlu_type} --random_split {str(args.random_split)}"

    elif args.type == "sar":
        log_path = f"logs/{dataset}/SAR/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/SAR.py --model {model} --dataset {dataset} --strategy {args.strategy} --temperature {str(args.temperature)} --device {args.gpu} --random_split {str(args.random_split)}"
    elif args.type == "sar_swarm":
        log_path = f"logs/{dataset}/SAR-swarm/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/SAR_swarm.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --random_split {str(args.random_split)}"
    elif args.type == "sar_mmlu":
        log_path = f"logs/{dataset}/SAR-MMLU/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/SAR_MMLU.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --type {args.mmlu_type} --random_split {str(args.random_split)}"
    elif args.type == "sar_time":
        log_path = f"logs/{dataset}/SAR/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/SAR_time.py --model {model} --dataset {dataset} --strategy {args.strategy} --temperature {str(args.temperature)} --device {args.gpu} --random_split {str(args.random_split)}"
    elif args.type == "sar_time_swarm":
        log_path = f"logs/{dataset}/SAR-swarm/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/SAR_swarm_time.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --random_split {str(args.random_split)}"
    elif args.type == "sar_time_mmlu":
        log_path = f"logs/{dataset}/SAR-MMLU/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/SAR_MMLU_time.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --type {args.mmlu_type} --random_split {str(args.random_split)}"

    
    elif args.type == "se":
        log_path = f"logs/{dataset}/SE/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/SE.py --model {model} --dataset {dataset} --strategy {args.strategy} --temperature {str(args.temperature)} --device {args.gpu} --random_split {str(args.random_split)}"
    elif args.type == "se_swarm":
        log_path = f"logs/{dataset}/SE-swarm/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/SE_swarm.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --random_split {str(args.random_split)}"
    elif args.type == "se_mmlu":
        log_path = f"logs/{dataset}/SE-MMLU/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/SE_MMLU.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --type {args.mmlu_type} --random_split {str(args.random_split)}"
    elif args.type == "se_time":
        log_path = f"logs/{dataset}/SE/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/SE_time.py --model {model} --dataset {dataset} --strategy {args.strategy} --temperature {str(args.temperature)} --device {args.gpu} --random_split {str(args.random_split)}"
    elif args.type == "se_time_swarm":
        log_path = f"logs/{dataset}/SE-swarm/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/SE_swarm_time.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --random_split {str(args.random_split)}"
    elif args.type == "se_time_mmlu":
        log_path = f"logs/{dataset}/SE-MMLU/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/SE_MMLU_time.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --type {args.mmlu_type} --random_split {str(args.random_split)}"
    

    elif args.type == "saup":
        log_path = f"logs/{dataset}/SAUP/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/SAUP.py --model {model} --dataset {dataset} --strategy {args.strategy} --temperature {str(args.temperature)} --device {args.gpu} --random_split {str(args.random_split)}"
    elif args.type == "saup_swarm":
        log_path = f"logs/{dataset}/SAUP-swarm/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/SAUP_swarm.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --random_split {str(args.random_split)}"
    elif args.type == "saup_mmlu":
        log_path = f"logs/{dataset}/SAUP-MMLU/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/SAUP_MMLU.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --type {args.mmlu_type} --random_split {str(args.random_split)}"
    elif args.type == "saup_time":
        log_path = f"logs/{dataset}/SAUP/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/SAUP_time.py --model {model} --dataset {dataset} --strategy {args.strategy} --temperature {str(args.temperature)} --device {args.gpu} --random_split {str(args.random_split)}"
    elif args.type == "saup_time_swarm":
        log_path = f"logs/{dataset}/SAUP-swarm/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/SAUP_swarm_time.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --random_split {str(args.random_split)}"
    elif args.type == "saup_time_mmlu":
        log_path = f"logs/{dataset}/SAUP-MMLU/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/SAUP_MMLU_time.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --type {args.mmlu_type} --random_split {str(args.random_split)}"
    

    elif args.type == "lars":
        log_path = f"logs/{dataset}/LARS/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/LARS.py --model {model} --dataset {dataset} --strategy {args.strategy} --temperature {str(args.temperature)} --device {args.gpu} --random_split {str(args.random_split)}"
    elif args.type == "lars_swarm":
        log_path = f"logs/{dataset}/LARS-swarm/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/LARS_swarm.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --random_split {str(args.random_split)}"
    elif args.type == "lars_mmlu":
        log_path = f"logs/{dataset}/LARS-MMLU/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/LARS_MMLU.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --type {args.mmlu_type} --random_split {str(args.random_split)}"
    elif args.type == "lars_time":
        log_path = f"logs/{dataset}/LARS/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/LARS_time.py --model {model} --dataset {dataset} --strategy {args.strategy} --temperature {str(args.temperature)} --device {args.gpu} --random_split {str(args.random_split)}"
    elif args.type == "lars_time_swarm":
        log_path = f"logs/{dataset}/LARS-swarm/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/LARS_swarm_time.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --random_split {str(args.random_split)}"
    elif args.type == "lars_time_mmlu":
        log_path = f"logs/{dataset}/LARS-MMLU/{dataset}-{model}-{args.strategy}-{str(int(10 * args.temperature))}.log"
        command=f"python UE/LARS_MMLU_time.py --model {model} --dataset {dataset} --agent_num {args.agent_num} --device {args.gpu} --type {args.mmlu_type} --random_split {str(args.random_split)}"
    else:
        print("Unkown type!")
        return
    
    parent_dir = os.path.dirname(log_path)
    os.makedirs(parent_dir, exist_ok=True)
    if os.path.exists(log_path):
        os.remove(log_path)
    
    print("\n\n")
    print("=" * 50 + "Running" + "=" * 50)
    print(command)
    print(flush=True)

    os.system(command)

    # import subprocess
    # import time
    # argument = '...'
    # proc = subprocess.Popen(['python', 'bar.py', argument], shell=True)
    # time.sleep(3) # <-- There's no time.wait, but time.sleep.
    # pid = proc.pid

task(args.dataset, args.models)
# processes = []
# for dataset in args.datasets:
#     for model in args.models:
#         p = Process(target=task, args=(dataset, model, ))
#         processes.append(p)
#         p.start()

# for p in processes:
#     p.join()





# python main_run.py --datasets HumanEval --model Qwen2.5-Coder-32B-Instruct --strategy SelfPlanningTestSnowballing --gpu 0,1 --type se --num_return_sequences 1
# python main_run.py --datasets HumanEval --model Qwen/Qwen2.5-Coder-7B-Instruct --strategy SelfPlanningTestSnowballing --gpu 2 --type sar --num_return_sequences 2
# python main_run.py --datasets HumanEval --model Qwen/Qwen3-14B --strategy SelfPlanningTestSnowballing --gpu 1 --type sar --num_return_sequences 2
# python main_run.py --datasets HumanEval --model Qwen/Qwen2.5-Coder-32B-Instruct --strategy SelfPlanningTestSnowballing --gpu 1，2 --type gen --num_return_sequences 2

# python main_run.py --datasets MBPP --model Qwen/Qwen3-30B-A3B-Instruct-2507 --strategy SelfPlanningTestSnowballing --gpu 0,1 --type gen --num_return_sequences 1
# python main_run.py --datasets MBPP --model Qwen/Qwen2.5-14B-Instruct --strategy SelfPlanningTestSnowballing --gpu 2 --type sar --num_return_sequences 2
# python main_run.py --datasets MBPP --model Qwen/Qwen3-14B --strategy SelfPlanningTestSnowballing --gpu 1 --type sar --num_return_sequences 2
# python main_run.py --datasets MBPP --model  Qwen/Qwen2.5-Coder-14B-Instruct --strategy SelfPlanningTestSnowballing --gpu 1 --type gen --num_return_sequences 2
# python main_run.py --datasets MBPP --model Qwen/Qwen2.5-Coder-3B-Instruct --strategy SelfPlanningTestSnowballing --gpu 3 --type gen --num_return_sequences 2

# python main_run.py --datasets MBPP --model Qwen/Qwen2.5-Coder-32B-Instruct --strategy SelfPlanningTestSnowballing --gpu 1,2 --type gen --num_return_sequences 2

# python main_run.py --datasets APPS --model Qwen/Qwen3-30B-A3B-Instruct-2507 --strategy SelfPlanningTestSnowballing --gpu 2,3 --type gen --num_return_sequences 1
# python main_run.py --datasets APPS --model Qwen/Qwen2.5-14B-Instruct --strategy SelfPlanningTestSnowballing --gpu 2 --type sar --num_return_sequences 2
# python main_run.py --datasets APPS --model Qwen/Qwen3-14B --strategy SelfPlanningTestSnowballing --gpu 1 --type sar --num_return_sequences 2
# python main_run.py --datasets APPS --model  Qwen/Qwen2.5-Coder-14B-Instruct --strategy SelfPlanningTestSnowballing --gpu 3 --type gen --num_return_sequences 2
# python main_run.py --datasets APPS --model Qwen/Qwen2.5-Coder-3B-Instruct --strategy SelfPlanningTestSnowballing --gpu 1 --type gen --num_return_sequences 2

# python main_run.py --datasets APPS --model Qwen/Qwen2.5-Coder-32B-Instruct --strategy SelfPlanningTestSnowballing --gpu 1,2 --type gen --num_return_sequences 2


# python main_run.py --datasets HumanEval --model Qwen/Qwen3-4B-Instruct-2507 -- strategy SelfPlanningTestSnowballing --gpu 1 --type saup
# python main_run.py --datasets MBPP --model Qwen/Qwen3-4B-Instruct-2507 --strategy SelfPlanningTestSnowballing --gpu 1 --type saup
# python main_run.py --datasets APPS --model Qwen/Qwen3-4B-Instruct-2507 --strategy SelfPlanningTestSnowballing --gpu 1 --type saup