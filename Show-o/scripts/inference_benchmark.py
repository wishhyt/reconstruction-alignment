#!/usr/bin/env python3
import subprocess
import os
import argparse
import os.path as osp

def generate_outdir_from_model_path(model_path, framework="geneval"):
    prefix = "out"
    if framework == "dpg":
        prefix = "dpg"
    elif framework == "wise":
        prefix = "wise"
        
    model_name = model_path.split('/')[-3]
    model_name = model_name.replace("show-o-reca-", "")
    model_name = model_name.replace("show-o-reca", "")
    model_name = model_name.replace("-", "_")
    
    if 'checkpoint-' in model_path:
        checkpoint = model_path.split('checkpoint-')[1].split('/')[0]
        checkpoint = checkpoint.replace("-", "_")
        return f"{prefix}_{model_name}_{checkpoint}"
    else:
        return f"{prefix}_{model_name}"

def main(model_path, framework="geneval", outdir=None, config_file="configs/showo_demo_512x512.yaml", 
         num_samples=None, real_gpu_list=[0, 1, 2, 3, 4, 5, 6, 7]):
    
    # Auto-determine num_samples based on framework if not specified
    if num_samples is None:
        num_samples = {
            'geneval': 12,
            'dpg': 4,
            'wise': 1,
        }.get(framework, 12)
    
    if outdir is None:
        outdir = generate_outdir_from_model_path(model_path, framework)
    
    script_name = f"{framework}.py"
    processes = []

    prompt_count = {
        'geneval': 560,
        'dpg': 1072,
        'wise': 1008,
    }[framework]
    
    for idx, gpu_id in enumerate(real_gpu_list):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        cmd = [
            "python3", script_name,
            f"config={config_file}",
            f"pretrained={model_path}",
            "batch_size=1",
            "mode=t2i",
            "guidance_scale=5",
            "generation_timesteps=50",
            f"num_samples={num_samples}",
            f"outdir={outdir}",
            f"l={idx * (prompt_count // len(real_gpu_list))}",
            f"r={(idx + 1) * (prompt_count // len(real_gpu_list))}",
        ]
        print(f"Starting process on GPU {gpu_id}: {' '.join(cmd)}")
        process = subprocess.Popen(cmd, env=env)
        processes.append(process)

    # Wait for all processes to complete
    for process in processes:
        process.wait()

    print(f"All {framework} processes have finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_paths", type=str, nargs='*', help="Model path(s), can be one or more, e.g. '/path/to/model1 /path/to/model2'")
    parser.add_argument("--model_path", type=str, dest="model_path_opt", help=argparse.SUPPRESS)
    parser.add_argument("--model_paths", type=str, nargs='+', dest="model_paths_opt", help="Multiple model paths, separated by spaces")
    parser.add_argument("--framework", type=str, choices=["geneval", "dpg", "wise"], default="geneval", 
                        help="Choose evaluation framework: geneval, dpg, or wise")
    parser.add_argument("--outdir", type=str, help="Output directory, auto-generated from model path if not specified")
    parser.add_argument("--config", type=str, default="configs/showo_demo_512x512.yaml", help="Config file path")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to generate per prompt (auto-detected by framework if not specified: geneval=12, dpg=4, wise=1)")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7", help="GPU device IDs to use, comma-separated, e.g. '0,1,2,3,4,5,6,7'")
    
    args = parser.parse_args()
    model_paths = []
    if args.model_paths:
        model_paths.extend(args.model_paths)
    if args.model_paths_opt:
        model_paths.extend(args.model_paths_opt)
    if args.model_path_opt:
        model_paths.append(args.model_path_opt)
    
    real_gpu_list = [int(gpu) for gpu in args.gpus.split(',')]
    if model_paths:
        print(model_paths)
        for model_path in model_paths:
            if model_path.endswith('/'):
                model_path = model_path[:-1]
            print(f"\n==== Processing model with {args.framework} framework: {model_path} ====")
            main(model_path, args.framework, args.outdir, args.config, args.num_samples, real_gpu_list)
    else:
        # If no command line arguments are provided, use default pretrained_list
        pretrained_list = []
        for pretrained in pretrained_list:
            main(pretrained[0], args.framework, pretrained[1], args.config, args.num_samples, real_gpu_list)