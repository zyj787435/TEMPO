import sys
import io
import os
# Add project root to sys.path so 'agentboard' is importable as a package
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# [AutoFix]  UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

import pdb
import sys
import os
import re
import warnings
import yaml
import json
import time
import argparse
from dotenv import load_dotenv
from tasks import load_task
from utils.logging.agent_logger import AgentLogger
# from utils.logging.logger import SummaryLogger
from autool.utils import PrintLogger

logger = AgentLogger(__name__)
warnings.filterwarnings("ignore")

TASKS=["alfworld", "jericho", "pddl", "webshop", "webarena", "tool-query", "tool-operation", "babyai", "scienceworld"]


def parse_args():
    parser = argparse.ArgumentParser(description="Testing")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--tasks", required=True, type=str, nargs='+',help="specify the tasks")
    parser.add_argument("--model", required=True ,help="specify the models, available models are stated in the configuration file")
    # parser.add_argument("--wandb", action="store_true", help="specify whether the wandb board is needed")
    parser.add_argument("--log_path", required=False, default='', help="specify the place to store the resuls")
    parser.add_argument("--project_name", required=False, default='', help="specify the project name for wandb")
    parser.add_argument("--baseline_dir", required=False, default='', help="specify the baseline loggings for wandb baseline comparison visualization")
    parser.add_argument("--max_num_steps", required=False, default=None, help="specify the maximum number of steps used to finish the problems")
    args = parser.parse_args()

    return args

def path_constructor(loader, node):
    path_matcher = re.compile(r'\$\{([^}^{]+)\}')
    ''' Extract the matched value, expand env variable, and replace the match '''
    value = node.value
    match = path_matcher.match(value)
    env_var = match.group()[2:-1]
    env_val = os.environ.get(env_var)
    if env_val is None:
        raise ValueError(f"Environment variable '${{{env_var}}}' used in config is not set. Please export {env_var} before running.")
    return env_val + value[match.end():]

def load_config(cfg_path, args):
    path_matcher = re.compile(r'\$\{([^}^{]+)\}')
    yaml.add_implicit_resolver('!path', path_matcher)
    yaml.add_constructor('!path', path_constructor)
    with open(cfg_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    llm_config = config["llm"]
    agent_config = config["agent"]
    env_config = config["env"]
    run_config = config["run"]
    
    if args.log_path != '':
        run_config["log_path"] = args.log_path
    if args.project_name != '':
        run_config["project_name"] = args.project_name
    if args.baseline_dir != '':
        run_config["baseline_dir"] = args.baseline_dir
        
    # run_config["wandb"] = args.wandb
    if args.max_num_steps is not None:
        run_config["max_num_steps"] = int(args.max_num_steps)

    return llm_config, agent_config, env_config, run_config
  
def check_log_paths_are_ready(log_dir, baseline_dir):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
        
    if not os.path.exists(os.path.join(log_dir, "logs")):
        os.makedirs(os.path.join(log_dir, "logs"))
    
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)
    
    if not os.path.exists(os.path.join(log_dir, 'all_results.txt')):
        with open(os.path.join(log_dir, 'all_results.txt'), "w") as f:
            f.write("")
            f.close()
            
    return True

def main():
    load_dotenv()  # take environment variables from .env., load openai api key, tool key, wandb key, project path...

    args = parse_args()
    llm_config, agent_config, env_config, run_config = load_config(args.cfg_path, args) 
    llm_config = llm_config[args.model]
    print(f"llm_config: {llm_config}")

    # Apply YAML llm config to autool.Config so call_model() uses correct endpoint/key/model
    try:
        from autool import config as _autool_cfg
        if llm_config.get("api_base"):
            _autool_cfg.Config.OPENAI_BASE_URL = llm_config["api_base"]
        if llm_config.get("api_key"):
            _autool_cfg.Config.OPENAI_API_KEY = llm_config["api_key"]
        if llm_config.get("engine"):
            _autool_cfg.Config.MODEL_NAME = llm_config["engine"]
        stop_val = llm_config.get("stop", None)
        _autool_cfg.Config.STOP = stop_val if stop_val else None
        print(f"[LLM Config Override] base_url={_autool_cfg.Config.OPENAI_BASE_URL}, model={_autool_cfg.Config.MODEL_NAME}, stop={repr(_autool_cfg.Config.STOP)}")
    except Exception as _e:
        print(f"[LLM Config Override] WARNING: failed to override Config: {_e}")

    #---------------------------------------------- load llm -----------------------------------------------------
    logger.info("Start loading language model")

    # llm = load_llm(llm_config["name"], llm_config)
    llm = None
    logger.info("Finished loading language model")
    
    
    
    #------------------------------------------------ initialize agentboard ------------------------------------
    # print(f'initializing agentboard')
    # if not run_config.get("wandb", False):
    #     os.environ["WANDB_MODE"] = "disabled"
    #     logger.info("Wandb is not enabled")
        
    #     wandb.init(mode="disabled")
    # else:
    #     wandb.init(
    #         project=run_config["project_name"],
    #         name="{model_name}_{engine_name}".format(model_name=llm_config["name"], engine_name=llm_config["engine"]),
    #         notes="An evaluation of LLM agent",
    #         config= {'llm_config': llm_config,
    #                     'agent_config': agent_config,
    #                     'env_config': env_config,
    #                     'run_config': run_config},
            
    #     )

    log_dir = run_config.get("log_path", None)
    

    baseline_path = run_config.get('baseline_dir', 'data/baseline_results_details')
    
    assert check_log_paths_are_ready(log_dir, baseline_path)
    
    # agentboard is the main launcher of visualizations and metrics calculation, 
    # agentboard = SummaryLogger(baseline_dir=baseline_path, log_path=log_dir) 
    
    log_history = dict()
    for line in open(os.path.join(log_dir, 'all_results.txt'), "r"):
        logger.info(line)
        if "_summary" not in line: 
            log_history[json.loads(line.strip())["task_name"]] = json.loads(line.strip())

    task_names = args.tasks if args.tasks != ["all"] else TASKS

    logger.info("Tested tasks: " + " ".join(log_history))
    print(f'[debug] log_history: {log_history}')
    #------------------------------------------------- start evaluation -------------------------------------------
    for task_name in task_names:
        print(f'evaluating task: {task_name}')
        # If the results of the task is already available at {log_path}/all_results.txt, skip the evaluation of this task to avoid rerunning. 
        # If you wish to rerun a task, make sure to remove the line recording previous task results from {log_path}/all_results.txt
        
        if task_name in log_history:
            logger.info(f"Task {task_name} has been evaluated, skip")
            
            
            # agentboard.log_run_result(task_name, log_history[task_name]["success_rate"], log_history[task_name]["progress_rate"], log_history[task_name]["grounding_acc"],
            #                           log_history[task_name]["success_rate_hard"], log_history[task_name]["success_rate_hard"], log_history[task_name]["progress_rate_hard"],
            #                           log_history[task_name]["progress_rate_easy"])

            continue
        
        logger.info(f"Start evaluating task {task_name}")

        agent_task_config = agent_config.copy()
        agent_task_config["task_type"] = task_name

        task_env_config = env_config[task_name]

        # Auto-select init_prompt_path from env.prompts dict based on agent name
        agent_name = agent_task_config.get("name", "ReactBaselineAgent")
        prompts_map = task_env_config.get("prompts", {})
        if prompts_map and agent_name in prompts_map:
            resolved_prompt = prompts_map[agent_name]
            agent_task_config["init_prompt_path"] = resolved_prompt
            task_env_config["init_prompt_path"] = resolved_prompt
            print(f"[Config] Auto-selected prompt for {agent_name}: {resolved_prompt}")
        elif "init_prompt_path" in task_env_config:
            agent_task_config["init_prompt_path"] = task_env_config["init_prompt_path"]

        # Copy env-level overrides to agent config
        for key in task_env_config:
            if key in ["check_actions", "check_inventory"]:
                agent_task_config[key] = task_env_config[key]

        # Per-task num_exam override
        task_run_config = run_config.copy()
        if "num_exam" in task_env_config:
            task_run_config["num_exam"] = task_env_config["num_exam"]

        if 'tool' in task_name:
            task = load_task('tool', task_run_config, llm_config, agent_task_config, task_env_config, llm=llm)
        else:
            task = load_task(task_name, task_run_config, llm_config, agent_task_config, task_env_config, llm=llm)

        print(f'load task done')
        success_rates, progress_rates, grounding_accs, score_state_records,\
            easy_sr, hard_sr, easy_pr, hard_pr = task.evaluate()
        
        print(f'evaluate done')
        
        # Check if lists are empty to avoid division by zero
        success_rate = sum(success_rates) * 1.0 / len(success_rates) if len(success_rates) > 0 else 0
        progress_rate = sum(progress_rates) * 1.0 / len(progress_rates) if len(progress_rates) > 0 else 0
        grounding_acc = sum(grounding_accs) * 1.0 / len(grounding_accs) if len(grounding_accs) > 0 else 0

        
        logger.finish(f"Task {task_name} | Success Rate: {success_rate} , Progress Rate: {progress_rate} , Easy SR: {easy_sr}."
                      f"Hard SR: {hard_sr}, Easy PR: {easy_pr}, Hard PR: {hard_pr}, Grounding Accuracy: {grounding_acc}")
        

        # agentboard.log_run_result(task_name, success_rate, progress_rate, grounding_acc, hard_sr, easy_sr, hard_pr, easy_pr)
        
            
    logger.info("Finish evaluating all tasks")
    
    
    # agentboard.log_summary()
        

if __name__ == "__main__":
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"agent_log_{time.strftime('%Y%m%d_%H%M%S')}.log")
    sys.stdout = PrintLogger(log_file)
    main()