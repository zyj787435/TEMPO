import jsonlines
from agentboard.common.registry import registry
from scienceworld import ScienceWorldEnv
import re

@registry.register_environment("scienceworld")
class Scienceworld:
    def __init__(self, serverPath=None, envStepLimit=100, label_path=''):
        self.env = ScienceWorldEnv("", serverPath, envStepLimit=envStepLimit)
        self.label_path = label_path
        self.labels = {}
        
        # 读取刚刚下载的 89 题文件
        with open(self.label_path, 'r+', encoding='utf-8') as f:
            for item in jsonlines.Reader(f):
                task_name = item["additional_info"]["env_name"]
                var = item["additional_info"]["var"]
                
                self.labels[f"{task_name}_{var}"] = {
                    "task_name": task_name,
                    "var": var,
                    "goal": item["goal"],
                    "modified_goal": item["goal"],
                    "subgoals": item.get("subgoals", []),
                    "difficulty": item["difficulty"],
                }

    def load(self, task_name, var, simplificationStr):
        try:
            # 尝试正常加载
            self.env.load(task_name, var, simplificationStr=simplificationStr)
        except ValueError as e:
            # ✅ [智能防崩] 遇到任务冲突，自动切换 easy 模式
            print(f"\n⚠️ [Auto-Fix] Task '{task_name}' conflict detected. Retrying with 'easy' mode...")
            self.env.load(task_name, var, simplificationStr='easy')

        self.cur_label = self.labels[f"{task_name}_{var}"]
        self.difficulty = self.cur_label["difficulty"] 
        self.max_score = 100 
        return self.env

    def inventory(self): return self.env.inventory()

    def step(self, action):
        action = action.strip()
        if action == "check valid actions":
            valid_actions = ", ".join(self.get_action_space(abstract=True))
            return f"Valid actions: {valid_actions}", 0.0, False, None

        # 使用原生分数
        observation, reward, done, info = self.env.step(action)
        current_score = info.get('score', 0)
        self.reward = current_score / 100.0
        self.done = done or (self.reward >= 1.0)
        return observation, self.reward, self.done, info

    def get_action_space(self, abstract=True):
        svalid_actions = []
        if abstract:
            for a in self.env.getPossibleActions():
                if "reset" not in a: svalid_actions.append(a)
        else:
            valid_actions = self.env.getValidActionObjectCombinationsWithTemplates()
            forbidden_words = ["teleport", "connect", "dunk", "eat", "flush", "close door"]
            for valid_action in valid_actions:
                v = valid_action['action']
                is_forbidden = False
                for fw in forbidden_words:
                    if fw in v: is_forbidden = True; break
                if not is_forbidden: svalid_actions.append(valid_action['action'])
        if "check valid actions" not in svalid_actions: svalid_actions.append("check valid actions")
        return svalid_actions

    def getTaskDescription(self): return self.env.getTaskDescription()
    def getGoalProgressStr(self): return self.env.getGoalProgressStr()
    def reset(self): self.reward = 0.; self.done = False; return self.env.reset()

    @classmethod
    def from_config(cls, cfg):
        return cls(serverPath=cfg.get("serverPath", None),
                   envStepLimit=cfg.get("envStepLimit", 50),
                   label_path=cfg.get("label_path", ''))
