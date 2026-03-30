"""
pddl_env.py — Text-based PDDL environment for 4 planning domains.

Domains: gripper, blockworld, barman, tyreworld
Each domain is a state-machine with ground predicates, action schemas,
text observations, and subgoal-based progress tracking.
"""

import json
import re
import copy
from pathlib import Path
from agentboard.common.registry import registry


# ═══════════════════════════════════════════════════════════════════════════
#  Domain Definitions
# ═══════════════════════════════════════════════════════════════════════════

class PDDLDomain:
    """Base class for PDDL domain simulation."""

    def __init__(self):
        self.state = set()       # set of ground predicate tuples
        self.goal = set()        # set of goal predicate tuples
        self.objects = {}        # type -> list of names
        self.actions = {}        # action_name -> (precond_fn, effect_fn)
        self.done = False
        self.won = False

    def get_obs(self):
        raise NotImplementedError

    def get_goal_text(self):
        raise NotImplementedError

    def get_valid_actions(self):
        raise NotImplementedError

    def apply_action(self, action_name, *args):
        raise NotImplementedError

    def check_goal(self):
        return self.goal.issubset(self.state)

    def get_progress(self):
        if not self.goal:
            return 0.0
        satisfied = sum(1 for g in self.goal if g in self.state)
        return satisfied / len(self.goal)


# ── Gripper Domain ──────────────────────────────────────────────────────

class GripperDomain(PDDLDomain):
    """Robot with two grippers moving balls between rooms."""

    def __init__(self, num_balls, ball_goals):
        """
        num_balls: total number of balls
        ball_goals: dict {ball_name: target_room}
        """
        super().__init__()
        rooms = ["rooma", "roomb"]
        grippers = ["left", "right"]
        balls = [f"ball{i}" for i in range(1, num_balls + 1)]

        self.objects = {"room": rooms, "gripper": grippers, "ball": balls}

        # Initial state: robot at rooma, all balls at rooma, grippers free
        self.state.add(("at-robby", "rooma"))
        for b in balls:
            self.state.add(("at", b, "rooma"))
        for g in grippers:
            self.state.add(("free", g))

        # Goal
        for ball_name, target_room in ball_goals.items():
            self.goal.add(("at", ball_name, target_room))

    def get_obs(self):
        # Robot location
        robby_loc = [a[1] for a in self.state if a[0] == "at-robby"][0]
        lines = [f"You are in {robby_loc}."]

        # Balls in current room
        balls_here = sorted([a[1] for a in self.state if a[0] == "at" and a[2] == robby_loc])
        if balls_here:
            lines.append(f"Balls here: {', '.join(balls_here)}.")
        else:
            lines.append("No balls here.")

        # Gripper status
        for g in self.objects["gripper"]:
            if ("free", g) in self.state:
                lines.append(f"Your {g} gripper is free.")
            else:
                carried = [a[1] for a in self.state if a[0] == "carry" and a[2] == g]
                if carried:
                    lines.append(f"Your {g} gripper holds {carried[0]}.")

        return "\n".join(lines)

    def get_goal_text(self):
        parts = []
        for g in sorted(self.goal, key=str):
            parts.append(f"{g[1]} is at {g[2]}")
        return "The goal is to satisfy the following conditions: " + ". ".join(parts) + "."

    def get_valid_actions(self):
        actions = []
        robby_loc = [a[1] for a in self.state if a[0] == "at-robby"][0]
        other_room = "roomb" if robby_loc == "rooma" else "rooma"

        # move
        actions.append(f"move {robby_loc} {other_room}")

        # pick
        balls_here = [a[1] for a in self.state if a[0] == "at" and a[2] == robby_loc]
        free_grippers = [a[1] for a in self.state if a[0] == "free"]
        for b in balls_here:
            for g in free_grippers:
                actions.append(f"pick {b} {robby_loc} {g}")

        # drop
        for g in self.objects["gripper"]:
            carried = [a[1] for a in self.state if a[0] == "carry" and a[2] == g]
            for b in carried:
                actions.append(f"drop {b} {robby_loc} {g}")

        return actions

    def apply_action(self, action_str):
        parts = action_str.strip().lower().split()
        if not parts:
            return False

        robby_loc = [a[1] for a in self.state if a[0] == "at-robby"][0]

        if parts[0] == "move" and len(parts) >= 3:
            from_r, to_r = parts[1], parts[2]
            if ("at-robby", from_r) in self.state:
                self.state.discard(("at-robby", from_r))
                self.state.add(("at-robby", to_r))
                return True

        elif parts[0] == "pick" and len(parts) >= 4:
            ball, room, gripper = parts[1], parts[2], parts[3]
            if (("at", ball, room) in self.state and
                ("at-robby", room) in self.state and
                ("free", gripper) in self.state):
                self.state.discard(("at", ball, room))
                self.state.discard(("free", gripper))
                self.state.add(("carry", ball, gripper))
                return True

        elif parts[0] == "drop" and len(parts) >= 4:
            ball, room, gripper = parts[1], parts[2], parts[3]
            if (("carry", ball, gripper) in self.state and
                ("at-robby", room) in self.state):
                self.state.discard(("carry", ball, gripper))
                self.state.add(("at", ball, room))
                self.state.add(("free", gripper))
                return True

        return False


# ── Blockworld Domain ───────────────────────────────────────────────────

class BlockworldDomain(PDDLDomain):
    """Classic blocks world: stack/unstack blocks on a table."""

    def __init__(self, num_blocks, block_goals, initial_config=None):
        """
        num_blocks: total blocks
        block_goals: list of ("on", bx, by) tuples
        initial_config: optional list of stacks, e.g. [["b1","b2"], ["b3"]]
        """
        super().__init__()
        blocks = [f"b{i}" for i in range(1, num_blocks + 1)]
        self.objects = {"block": blocks}

        # Default initial state: all blocks on table, all clear, arm empty
        self.state.add(("arm-empty",))
        if initial_config:
            for stack in initial_config:
                for i, b in enumerate(stack):
                    if i == 0:
                        self.state.add(("on-table", b))
                    else:
                        self.state.add(("on", b, stack[i-1]))
                    if i == len(stack) - 1:
                        self.state.add(("clear", b))
        else:
            for b in blocks:
                self.state.add(("on-table", b))
                self.state.add(("clear", b))

        # Goal
        for g in block_goals:
            self.goal.add(g)

    def get_obs(self):
        lines = []
        # Describe stacks
        on_table = sorted([a[1] for a in self.state if a[0] == "on-table"])
        for base in on_table:
            stack = [base]
            current = base
            while True:
                above = [a[1] for a in self.state if a[0] == "on" and a[2] == current]
                if above:
                    stack.append(above[0])
                    current = above[0]
                else:
                    break
            lines.append(f"Stack: {' -> '.join(stack)} (top: {stack[-1]})")

        holding = [a[1] for a in self.state if a[0] == "holding"]
        if holding:
            lines.append(f"You are holding {holding[0]}.")
        else:
            lines.append("Your arm is empty.")

        clear_blocks = sorted([a[1] for a in self.state if a[0] == "clear"])
        if clear_blocks:
            lines.append(f"Clear blocks: {', '.join(clear_blocks)}.")

        return "\n".join(lines)

    def get_goal_text(self):
        parts = []
        for g in sorted(self.goal, key=str):
            if g[0] == "on":
                parts.append(f"{g[1]} is on {g[2]}")
            elif g[0] == "on-table":
                parts.append(f"{g[1]} is on the table")
        return "The goal is to satisfy the following conditions: " + ". ".join(parts) + "."

    def get_valid_actions(self):
        actions = []
        clear_blocks = [a[1] for a in self.state if a[0] == "clear"]
        arm_empty = ("arm-empty",) in self.state
        holding = [a[1] for a in self.state if a[0] == "holding"]

        if arm_empty:
            for b in clear_blocks:
                if ("on-table", b) in self.state:
                    actions.append(f"pick-up {b}")
                on_below = [a[2] for a in self.state if a[0] == "on" and a[1] == b]
                if on_below:
                    actions.append(f"unstack {b} {on_below[0]}")
        else:
            b = holding[0]
            actions.append(f"put-down {b}")
            for c in clear_blocks:
                actions.append(f"stack {b} {c}")

        return actions

    def apply_action(self, action_str):
        parts = action_str.strip().lower().split()
        if not parts:
            return False

        if parts[0] == "pick-up" and len(parts) >= 2:
            b = parts[1]
            if (("clear", b) in self.state and
                ("on-table", b) in self.state and
                ("arm-empty",) in self.state):
                self.state.discard(("on-table", b))
                self.state.discard(("clear", b))
                self.state.discard(("arm-empty",))
                self.state.add(("holding", b))
                return True

        elif parts[0] == "put-down" and len(parts) >= 2:
            b = parts[1]
            if ("holding", b) in self.state:
                self.state.discard(("holding", b))
                self.state.add(("on-table", b))
                self.state.add(("clear", b))
                self.state.add(("arm-empty",))
                return True

        elif parts[0] == "stack" and len(parts) >= 3:
            b, c = parts[1], parts[2]
            if (("holding", b) in self.state and
                ("clear", c) in self.state):
                self.state.discard(("holding", b))
                self.state.discard(("clear", c))
                self.state.add(("on", b, c))
                self.state.add(("clear", b))
                self.state.add(("arm-empty",))
                return True

        elif parts[0] == "unstack" and len(parts) >= 3:
            b, c = parts[1], parts[2]
            if (("on", b, c) in self.state and
                ("clear", b) in self.state and
                ("arm-empty",) in self.state):
                self.state.discard(("on", b, c))
                self.state.discard(("clear", b))
                self.state.discard(("arm-empty",))
                self.state.add(("holding", b))
                self.state.add(("clear", c))
                return True

        return False


# ── Barman Domain ───────────────────────────────────────────────────────

class BarmanDomain(PDDLDomain):
    """Bartender mixing cocktails from ingredients using shaker and shots."""

    def __init__(self, num_cocktails, num_ingredients, num_shots, shot_goals):
        super().__init__()
        self.num_cocktails = num_cocktails
        self.num_ingredients = num_ingredients

        cocktails = [f"cocktail{i}" for i in range(1, num_cocktails + 1)]
        ingredients = [f"ingredient{i}" for i in range(1, num_ingredients + 1)]
        shots = [f"shot{i}" for i in range(1, num_shots + 1)]
        dispensers = [f"dispenser{i}" for i in range(1, num_ingredients + 1)]

        self.objects = {
            "cocktail": cocktails, "ingredient": ingredients,
            "shot": shots, "dispenser": dispensers,
            "hand": ["left", "right"], "shaker": ["shaker1"],
            "level": ["l0", "l1", "l2"]
        }

        # Cocktail recipes: cocktail_i = ingredient_i + ingredient_(i%n)+1
        self.recipes = {}
        for i in range(1, num_cocktails + 1):
            ing1 = f"ingredient{((i-1) % num_ingredients) + 1}"
            ing2 = f"ingredient{(i % num_ingredients) + 1}"
            self.recipes[f"cocktail{i}"] = (ing1, ing2)

        # Dispenser mapping
        self.dispenser_map = {}
        for i in range(1, num_ingredients + 1):
            self.dispenser_map[f"dispenser{i}"] = f"ingredient{i}"

        # Initial state
        for h in ["left", "right"]:
            self.state.add(("hand-empty", h))
        for s in shots:
            self.state.add(("clean", s))
            self.state.add(("empty", s))
            self.state.add(("on-table", s))
        self.state.add(("clean", "shaker1"))
        self.state.add(("empty", "shaker1"))
        self.state.add(("on-table", "shaker1"))
        self.state.add(("shaker-level", "shaker1", "l0"))

        # Goal
        for shot_name, content in shot_goals.items():
            self.goal.add(("contains", shot_name, content))

    def get_obs(self):
        lines = []
        # Hands
        for h in ["left", "right"]:
            if ("hand-empty", h) in self.state:
                lines.append(f"Your {h} hand is empty.")
            else:
                held = [a[2] for a in self.state if a[0] == "holding" and a[1] == h]
                if held:
                    lines.append(f"Your {h} hand holds {held[0]}.")

        # Shots
        for s in sorted(self.objects["shot"]):
            parts = []
            if ("clean", s) in self.state:
                parts.append("clean")
            if ("empty", s) in self.state:
                parts.append("empty")
            contains = [a[2] for a in self.state if a[0] == "contains" and a[1] == s]
            if contains:
                parts.append(f"contains {contains[0]}")
            if ("on-table", s) in self.state:
                parts.append("on table")
            lines.append(f"{s}: {', '.join(parts)}.")

        # Shaker
        s = "shaker1"
        parts = []
        if ("clean", s) in self.state:
            parts.append("clean")
        if ("empty", s) in self.state:
            parts.append("empty")
        shaker_contains = [a[2] for a in self.state if a[0] == "shaker-contains" and a[1] == s]
        if shaker_contains:
            parts.append(f"contains {', '.join(shaker_contains)}")
        level = [a[2] for a in self.state if a[0] == "shaker-level" and a[1] == s]
        if level:
            parts.append(f"level {level[0]}")
        if ("shaked", s) in self.state:
            parts.append("shaked")
        lines.append(f"shaker1: {', '.join(parts)}.")

        # Dispensers
        lines.append("Dispensers: " + ", ".join(
            f"{d} (has {self.dispenser_map[d]})" for d in sorted(self.objects["dispenser"])))

        # Recipes
        lines.append("Recipes: " + "; ".join(
            f"{c} = {r[0]} + {r[1]}" for c, r in sorted(self.recipes.items())))

        return "\n".join(lines)

    def get_goal_text(self):
        parts = []
        for g in sorted(self.goal, key=str):
            parts.append(f"{g[1]} contains {g[2]}")
        return "The goal is to satisfy the following conditions: " + ". ".join(parts) + "."

    def get_valid_actions(self):
        actions = []
        hands = ["left", "right"]
        empty_hands = [h for h in hands if ("hand-empty", h) in self.state]
        held_items = {a[1]: a[2] for a in self.state if a[0] == "holding"}

        # grasp / leave
        for h in empty_hands:
            for s in self.objects["shot"] + ["shaker1"]:
                if ("on-table", s) in self.state:
                    actions.append(f"grasp {h} {s}")
        for h, item in held_items.items():
            actions.append(f"leave {h} {item}")

        # fill-shot: fill shot with ingredient from dispenser
        for h_hold, item in held_items.items():
            if item.startswith("shot"):
                if ("empty", item) in self.state and ("clean", item) in self.state:
                    other_hands = [h for h in empty_hands if h != h_hold]
                    # can also use a hand holding nothing else
                    for d in self.objects["dispenser"]:
                        ing = self.dispenser_map[d]
                        actions.append(f"fill-shot {item} {ing} {h_hold} {d}")

        # empty-shot
        for h, item in held_items.items():
            if item.startswith("shot"):
                contains = [a[2] for a in self.state if a[0] == "contains" and a[1] == item]
                if contains:
                    actions.append(f"empty-shot {h} {item} {contains[0]}")

        # clean-shot
        for h, item in held_items.items():
            if item.startswith("shot"):
                if ("empty", item) in self.state and ("clean", item) not in self.state:
                    other_empty = [h2 for h2 in empty_hands if h2 != h]
                    if other_empty:
                        actions.append(f"clean-shot {item} {h} {other_empty[0]}")

        # pour-shot-to-shaker
        for h, item in held_items.items():
            if item.startswith("shot"):
                contains = [a[2] for a in self.state if a[0] == "contains" and a[1] == item]
                if contains and ("shaked", "shaker1") not in self.state:
                    cur_level = [a[2] for a in self.state if a[0] == "shaker-level" and a[1] == "shaker1"]
                    if cur_level and cur_level[0] != "l2":
                        actions.append(f"pour-shot-to-shaker {item} {contains[0]} shaker1")

        # shake
        if ("shaked", "shaker1") not in self.state:
            shaker_contents = [a[2] for a in self.state if a[0] == "shaker-contains" and a[1] == "shaker1"]
            if len(shaker_contents) >= 2:
                # Check if both hands are used (one holding shaker)
                for h, item in held_items.items():
                    if item == "shaker1":
                        other_empty = [h2 for h2 in empty_hands]
                        # Need the other hand empty or not — simplified: just need to hold shaker
                        actions.append(f"shake shaker1")

        # pour-shaker-to-shot
        if ("shaked", "shaker1") in self.state:
            for h, item in held_items.items():
                if item == "shaker1":
                    cocktail = [a[2] for a in self.state if a[0] == "shaker-contains" and a[1] == "shaker1"]
                    # Find which cocktail was made
                    for c_name, (i1, i2) in self.recipes.items():
                        if i1 in cocktail and i2 in cocktail:
                            for s in self.objects["shot"]:
                                if ("empty", s) in self.state and ("clean", s) in self.state:
                                    actions.append(f"pour-shaker-to-shot {c_name} shaker1 {s}")

        # empty-shaker / clean-shaker
        for h, item in held_items.items():
            if item == "shaker1":
                if ("empty", "shaker1") not in self.state:
                    actions.append(f"empty-shaker {h} shaker1")
                if ("empty", "shaker1") in self.state and ("clean", "shaker1") not in self.state:
                    other_empty = [h2 for h2 in empty_hands if h2 != h]
                    if other_empty:
                        actions.append(f"clean-shaker shaker1 {h} {other_empty[0]}")

        if not actions:
            actions.append("check valid actions")

        return actions

    def apply_action(self, action_str):
        parts = action_str.strip().lower().split()
        if not parts:
            return False
        cmd = parts[0]

        if cmd == "grasp" and len(parts) >= 3:
            hand, item = parts[1], parts[2]
            if ("hand-empty", hand) in self.state and ("on-table", item) in self.state:
                self.state.discard(("hand-empty", hand))
                self.state.discard(("on-table", item))
                self.state.add(("holding", hand, item))
                return True

        elif cmd == "leave" and len(parts) >= 3:
            hand, item = parts[1], parts[2]
            if ("holding", hand, item) in self.state:
                self.state.discard(("holding", hand, item))
                self.state.add(("hand-empty", hand))
                self.state.add(("on-table", item))
                return True

        elif cmd == "fill-shot" and len(parts) >= 5:
            shot, ingredient, hand, dispenser = parts[1], parts[2], parts[3], parts[4]
            if (("holding", hand, shot) in self.state and
                ("empty", shot) in self.state and
                ("clean", shot) in self.state and
                self.dispenser_map.get(dispenser) == ingredient):
                self.state.discard(("empty", shot))
                self.state.discard(("clean", shot))
                self.state.add(("contains", shot, ingredient))
                return True

        elif cmd == "empty-shot" and len(parts) >= 4:
            hand, shot, beverage = parts[1], parts[2], parts[3]
            if (("holding", hand, shot) in self.state and
                ("contains", shot, beverage) in self.state):
                self.state.discard(("contains", shot, beverage))
                self.state.add(("empty", shot))
                return True

        elif cmd == "clean-shot" and len(parts) >= 4:
            shot, hand1, hand2 = parts[1], parts[2], parts[3]
            if (("holding", hand1, shot) in self.state and
                ("empty", shot) in self.state and
                ("hand-empty", hand2) in self.state):
                self.state.add(("clean", shot))
                return True

        elif cmd == "pour-shot-to-shaker" and len(parts) >= 4:
            shot, ingredient, shaker = parts[1], parts[2], parts[3]
            if ("contains", shot, ingredient) in self.state:
                self.state.discard(("contains", shot, ingredient))
                self.state.add(("empty", shot))
                self.state.add(("shaker-contains", shaker, ingredient))
                # Update level
                cur = [a[2] for a in self.state if a[0] == "shaker-level" and a[1] == shaker]
                if cur:
                    self.state.discard(("shaker-level", shaker, cur[0]))
                    next_l = "l1" if cur[0] == "l0" else "l2"
                    self.state.discard(("empty", shaker))
                    self.state.add(("shaker-level", shaker, next_l))
                return True

        elif cmd == "shake" and len(parts) >= 2:
            shaker = parts[1]
            contents = [a[2] for a in self.state if a[0] == "shaker-contains" and a[1] == shaker]
            if len(contents) >= 2:
                self.state.add(("shaked", shaker))
                return True

        elif cmd == "pour-shaker-to-shot" and len(parts) >= 4:
            cocktail, shaker, shot = parts[1], parts[2], parts[3]
            if (("shaked", shaker) in self.state and
                ("empty", shot) in self.state and
                ("clean", shot) in self.state):
                self.state.add(("contains", shot, cocktail))
                self.state.discard(("empty", shot))
                self.state.discard(("clean", shot))
                return True

        elif cmd == "empty-shaker" and len(parts) >= 3:
            hand, shaker = parts[1], parts[2]
            if ("holding", hand, shaker) in self.state:
                # Clear shaker contents
                to_remove = [a for a in self.state if a[0] == "shaker-contains" and a[1] == shaker]
                for a in to_remove:
                    self.state.discard(a)
                self.state.discard(("shaked", shaker))
                self.state.add(("empty", shaker))
                # Reset level
                cur = [a for a in self.state if a[0] == "shaker-level" and a[1] == shaker]
                for a in cur:
                    self.state.discard(a)
                self.state.add(("shaker-level", shaker, "l0"))
                return True

        elif cmd == "clean-shaker" and len(parts) >= 4:
            shaker, hand1, hand2 = parts[1], parts[2], parts[3]
            if (("holding", hand1, shaker) in self.state and
                ("empty", shaker) in self.state and
                ("hand-empty", hand2) in self.state):
                self.state.add(("clean", shaker))
                return True

        return False


# ── Tyreworld Domain ────────────────────────────────────────────────────

class TyreworldDomain(PDDLDomain):
    """Changing flat tyres: jack up hub, remove old wheel, put on spare, inflate."""

    def __init__(self, num_hubs, hub_goals):
        """
        num_hubs: number of hubs
        hub_goals: dict describing target state
        """
        super().__init__()
        hubs = [f"the-hub{i}" for i in range(1, num_hubs + 1)]
        nuts = [f"nut{i}" for i in range(1, num_hubs + 1)]
        old_wheels = [f"w{i}" for i in range(1, num_hubs + 1)]
        new_wheels = [f"r{i}" for i in range(1, num_hubs + 1)]

        self.objects = {
            "hub": hubs, "nut": nuts,
            "wheel": old_wheels + new_wheels,
            "container": ["boot"],
            "tool": ["wrench", "jack", "pump"]
        }

        # Initial state
        self.state.add(("have", "wrench"))
        self.state.add(("have", "jack"))
        self.state.add(("have", "pump"))
        self.state.add(("unlocked", "boot"))

        for i in range(1, num_hubs + 1):
            hub = f"the-hub{i}"
            nut = f"nut{i}"
            old_w = f"w{i}"
            new_w = f"r{i}"
            # Old wheel on hub, tight nut, hub on ground
            self.state.add(("on", old_w, hub))
            self.state.add(("on-ground", hub))
            self.state.add(("tight", nut, hub))
            self.state.add(("inflated", old_w))
            self.state.add(("intact", old_w))
            # New wheel in boot, not inflated
            self.state.add(("in", new_w, "boot"))
            self.state.add(("intact", new_w))

        # Goal
        for pred in hub_goals:
            self.goal.add(pred)

    def get_obs(self):
        lines = []
        # Hubs
        for hub in sorted(self.objects["hub"]):
            parts = []
            wheel_on = [a[1] for a in self.state if a[0] == "on" and a[2] == hub]
            if wheel_on:
                parts.append(f"{wheel_on[0]} is on {hub}")
            nut_status = []
            for a in self.state:
                if a[0] == "tight" and a[2] == hub:
                    nut_status.append(f"{a[1]} is tight")
                elif a[0] == "loose" and a[2] == hub:
                    nut_status.append(f"{a[1]} is loose")
            parts.extend(nut_status)
            if ("on-ground", hub) in self.state:
                parts.append("on ground")
            elif ("jacked-up", hub) in self.state:
                parts.append("jacked up")
            lines.append(f"{hub}: {', '.join(parts)}.")

        # Boot
        in_boot = sorted([a[1] for a in self.state if a[0] == "in" and a[2] == "boot"])
        if in_boot:
            lines.append(f"In boot: {', '.join(in_boot)}.")
        else:
            lines.append("Boot is empty.")

        # Held items
        held = [a[1] for a in self.state if a[0] == "have"]
        tools_held = [h for h in held if h in ["wrench", "jack", "pump"]]
        other_held = [h for h in held if h not in ["wrench", "jack", "pump"]]
        if tools_held:
            lines.append(f"Tools: {', '.join(sorted(tools_held))}.")
        if other_held:
            lines.append(f"Holding: {', '.join(sorted(other_held))}.")

        # Wheel inflation status
        inflated = sorted([a[1] for a in self.state if a[0] == "inflated"])
        if inflated:
            lines.append(f"Inflated wheels: {', '.join(inflated)}.")

        return "\n".join(lines)

    def get_goal_text(self):
        parts = []
        for g in sorted(self.goal, key=str):
            if g[0] == "on":
                parts.append(f"{g[1]} is on {g[2]}")
            elif g[0] == "in":
                parts.append(f"{g[1]} is in {g[2]}")
            elif g[0] == "inflated":
                parts.append(f"Wheel {g[1]} is inflated")
        return "The goal is to satisfy the following conditions: " + ". ".join(parts) + "."

    def get_valid_actions(self):
        actions = []
        for hub in self.objects["hub"]:
            nut = hub.replace("the-hub", "nut")

            # loosen / tighten
            if ("tight", nut, hub) in self.state and ("on-ground", hub) in self.state and ("have", "wrench") in self.state:
                actions.append(f"loosen {nut} {hub}")
            if ("loose", nut, hub) in self.state and ("on-ground", hub) in self.state and ("have", "wrench") in self.state:
                actions.append(f"tighten {nut} {hub}")

            # jack-up / jack-down
            if ("on-ground", hub) in self.state and ("have", "jack") in self.state:
                actions.append(f"jack-up {hub}")
            if ("jacked-up", hub) in self.state:
                actions.append(f"jack-down {hub}")

            # undo nut (remove nut from hub when jacked and loose)
            if ("jacked-up", hub) in self.state and ("loose", nut, hub) in self.state and ("have", "wrench") in self.state:
                actions.append(f"undo {nut} {hub}")
            # do-up nut
            if ("jacked-up", hub) in self.state and ("have", nut) in self.state and ("have", "wrench") in self.state:
                unfastened = not any(a[0] in ("tight", "loose") and a[1] == nut for a in self.state)
                if unfastened:
                    actions.append(f"do-up {nut} {hub}")

            # remove-wheel
            wheel_on = [a[1] for a in self.state if a[0] == "on" and a[2] == hub]
            if wheel_on and ("jacked-up", hub) in self.state:
                nut_off = not any(a[0] in ("tight", "loose") and a[2] == hub for a in self.state)
                if nut_off:
                    actions.append(f"remove-wheel {wheel_on[0]} {hub}")

            # put-on-wheel
            if ("jacked-up", hub) in self.state and not wheel_on:
                nut_off = not any(a[0] in ("tight", "loose") and a[2] == hub for a in self.state)
                if nut_off:
                    have_wheels = [a[1] for a in self.state if a[0] == "have" and a[1].startswith(("w", "r"))]
                    for w in have_wheels:
                        actions.append(f"put-on-wheel {w} {hub}")

        # inflate
        for w in self.objects["wheel"]:
            if ("have", w) in self.state and ("inflated", w) not in self.state and ("intact", w) in self.state and ("have", "pump") in self.state:
                actions.append(f"inflate {w}")
            # Also can inflate if on hub (not held)
            on_hub = [a[2] for a in self.state if a[0] == "on" and a[1] == w]
            if on_hub and ("inflated", w) not in self.state and ("intact", w) in self.state and ("have", "pump") in self.state:
                actions.append(f"inflate {w}")

        # put-away / fetch from boot
        for w in self.objects["wheel"]:
            if ("have", w) in self.state and ("unlocked", "boot") in self.state:
                actions.append(f"put-away {w} boot")
        in_boot = [a[1] for a in self.state if a[0] == "in" and a[2] == "boot"]
        for item in in_boot:
            if ("unlocked", "boot") in self.state:
                actions.append(f"fetch {item} boot")

        if not actions:
            actions.append("check valid actions")

        return actions

    def apply_action(self, action_str):
        parts = action_str.strip().lower().split()
        if not parts:
            return False
        cmd = parts[0]

        if cmd == "loosen" and len(parts) >= 3:
            nut, hub = parts[1], parts[2]
            if ("tight", nut, hub) in self.state and ("on-ground", hub) in self.state and ("have", "wrench") in self.state:
                self.state.discard(("tight", nut, hub))
                self.state.add(("loose", nut, hub))
                return True

        elif cmd == "tighten" and len(parts) >= 3:
            nut, hub = parts[1], parts[2]
            if ("loose", nut, hub) in self.state and ("on-ground", hub) in self.state and ("have", "wrench") in self.state:
                self.state.discard(("loose", nut, hub))
                self.state.add(("tight", nut, hub))
                return True

        elif cmd == "jack-up" and len(parts) >= 2:
            hub = parts[1]
            if ("on-ground", hub) in self.state and ("have", "jack") in self.state:
                self.state.discard(("on-ground", hub))
                self.state.discard(("have", "jack"))
                self.state.add(("jacked-up", hub))
                return True

        elif cmd == "jack-down" and len(parts) >= 2:
            hub = parts[1]
            if ("jacked-up", hub) in self.state:
                self.state.discard(("jacked-up", hub))
                self.state.add(("on-ground", hub))
                self.state.add(("have", "jack"))
                return True

        elif cmd == "undo" and len(parts) >= 3:
            nut, hub = parts[1], parts[2]
            if ("jacked-up", hub) in self.state and ("loose", nut, hub) in self.state and ("have", "wrench") in self.state:
                self.state.discard(("loose", nut, hub))
                self.state.add(("have", nut))
                return True

        elif cmd == "do-up" and len(parts) >= 3:
            nut, hub = parts[1], parts[2]
            if ("jacked-up", hub) in self.state and ("have", nut) in self.state and ("have", "wrench") in self.state:
                self.state.discard(("have", nut))
                self.state.add(("loose", nut, hub))
                return True

        elif cmd == "remove-wheel" and len(parts) >= 3:
            wheel, hub = parts[1], parts[2]
            if ("on", wheel, hub) in self.state and ("jacked-up", hub) in self.state:
                self.state.discard(("on", wheel, hub))
                self.state.add(("have", wheel))
                return True

        elif cmd == "put-on-wheel" and len(parts) >= 3:
            wheel, hub = parts[1], parts[2]
            if ("have", wheel) in self.state and ("jacked-up", hub) in self.state:
                wheel_on = [a for a in self.state if a[0] == "on" and a[2] == hub]
                if not wheel_on:
                    self.state.discard(("have", wheel))
                    self.state.add(("on", wheel, hub))
                    return True

        elif cmd == "inflate" and len(parts) >= 2:
            wheel = parts[1]
            if ("intact", wheel) in self.state and ("inflated", wheel) not in self.state and ("have", "pump") in self.state:
                self.state.add(("inflated", wheel))
                return True

        elif cmd == "put-away" and len(parts) >= 3:
            item, container = parts[1], parts[2]
            if ("have", item) in self.state and ("unlocked", container) in self.state:
                self.state.discard(("have", item))
                self.state.add(("in", item, container))
                return True

        elif cmd == "fetch" and len(parts) >= 3:
            item, container = parts[1], parts[2]
            if ("in", item, container) in self.state and ("unlocked", container) in self.state:
                self.state.discard(("in", item, container))
                self.state.add(("have", item))
                return True

        return False


# ═══════════════════════════════════════════════════════════════════════════
#  Problem Generator — creates domain instances from test.jsonl goals
# ═══════════════════════════════════════════════════════════════════════════

def _parse_gripper_goals(goal_text):
    """Parse 'ball1 is at roomb. ball2 is at rooma.' into dict."""
    goals = {}
    for m in re.finditer(r'(ball\d+)\s+is at\s+(room[ab])', goal_text):
        goals[m.group(1)] = m.group(2)
    return goals

def _parse_blockworld_goals(goal_text):
    """Parse 'b2 is on b1. b3 is on b2.' into list of on-tuples."""
    goals = []
    for m in re.finditer(r'(b\d+)\s+is on\s+(b\d+)', goal_text):
        goals.append(("on", m.group(1), m.group(2)))
    return goals

def _parse_barman_goals(goal_text):
    """Parse 'shot1 contains cocktail1. shot2 contains ingredient2.'"""
    goals = {}
    for m in re.finditer(r'(shot\d+)\s+contains\s+((?:cocktail|ingredient)\d+)', goal_text):
        goals[m.group(1)] = m.group(2)
    return goals

def _parse_tyreworld_goals(goal_text):
    """Parse tyreworld goals into predicate tuples."""
    goals = []
    for m in re.finditer(r'Wheel\s+(r\d+)\s+is inflated', goal_text):
        goals.append(("inflated", m.group(1)))
    for m in re.finditer(r'(r\d+)\s+is on\s+(the-hub\d+)', goal_text):
        goals.append(("on", m.group(1), m.group(2)))
    for m in re.finditer(r'(w\d+)\s+is in\s+boot', goal_text):
        goals.append(("in", m.group(1), "boot"))
    return goals


def create_domain(subtask, goal_text, problem_index):
    """Create a domain instance from test.jsonl entry."""
    if subtask == "gripper":
        ball_goals = _parse_gripper_goals(goal_text)
        all_balls = set(ball_goals.keys())
        num_balls = max(int(b.replace("ball", "")) for b in all_balls) if all_balls else 4
        return GripperDomain(num_balls, ball_goals)

    elif subtask == "blockworld":
        block_goals = _parse_blockworld_goals(goal_text)
        all_blocks = set()
        for g in block_goals:
            all_blocks.add(g[1])
            all_blocks.add(g[2])
        num_blocks = max(int(b.replace("b", "")) for b in all_blocks)
        # Generate a random but solvable initial config (all on table)
        return BlockworldDomain(num_blocks, block_goals)

    elif subtask == "barman":
        shot_goals = _parse_barman_goals(goal_text)
        all_contents = set(shot_goals.values())
        max_cocktail = 0
        max_ingredient = 0
        for c in all_contents:
            if c.startswith("cocktail"):
                max_cocktail = max(max_cocktail, int(c.replace("cocktail", "")))
            elif c.startswith("ingredient"):
                max_ingredient = max(max_ingredient, int(c.replace("ingredient", "")))
        num_cocktails = max(max_cocktail, 3)
        num_ingredients = max(max_ingredient, 3)
        max_shot = max(int(s.replace("shot", "")) for s in shot_goals.keys())
        num_shots = max(max_shot, 3)
        return BarmanDomain(num_cocktails, num_ingredients, num_shots, shot_goals)

    elif subtask == "tyreworld":
        hub_goals = _parse_tyreworld_goals(goal_text)
        # Determine number of hubs from goals
        all_hub_nums = set()
        for g in hub_goals:
            if g[0] == "on" and "hub" in g[2]:
                all_hub_nums.add(int(g[2].replace("the-hub", "")))
            elif g[0] == "inflated":
                all_hub_nums.add(int(g[1].replace("r", "")))
            elif g[0] == "in":
                all_hub_nums.add(int(g[1].replace("w", "")))
        num_hubs = max(all_hub_nums) if all_hub_nums else 1
        return TyreworldDomain(num_hubs, hub_goals)

    else:
        raise ValueError(f"Unknown PDDL subtask: {subtask}")


# ═══════════════════════════════════════════════════════════════════════════
#  Registered Environment Wrapper
# ═══════════════════════════════════════════════════════════════════════════

@registry.register_environment("pddl")
class PDDLEnv:
    """Wrapper that presents PDDL domains as a text-based environment."""

    def __init__(self, game_name, problem_index, difficulty="easy",
                 label_path=None, **kwargs):
        self.game_name = game_name
        self.problem_index = problem_index
        self.difficulty = difficulty
        self.label_path = label_path
        self.domain = None
        self.won = False
        self._setup()

    def _setup(self):
        """Load the problem from test.jsonl and create domain."""
        if self.label_path:
            with open(self.label_path) as f:
                all_tasks = [json.loads(l) for l in f if l.strip()]

            # Find the matching task
            idx = 0
            for t in all_tasks:
                if t["additional_info"]["subtask"] == self.game_name:
                    if idx == self.problem_index:
                        self.domain = create_domain(
                            self.game_name,
                            t["goal"],
                            self.problem_index
                        )
                        self.difficulty = t.get("difficulty", self.difficulty)
                        return
                    idx += 1

        # Fallback: create a simple default problem
        if self.domain is None:
            if self.game_name == "gripper":
                self.domain = GripperDomain(4, {"ball1": "roomb", "ball2": "roomb"})
            elif self.game_name == "blockworld":
                self.domain = BlockworldDomain(3, [("on", "b2", "b1"), ("on", "b3", "b2")])
            elif self.game_name == "barman":
                self.domain = BarmanDomain(3, 3, 3, {"shot1": "cocktail1"})
            elif self.game_name == "tyreworld":
                self.domain = TyreworldDomain(1, [("inflated", "r1"), ("on", "r1", "the-hub1"), ("in", "w1", "boot")])

    def _get_obs(self):
        return self.domain.get_obs()

    def _get_goal(self):
        return self.domain.get_goal_text()

    def get_action_space(self):
        return self.domain.get_valid_actions()

    def step(self, action):
        action = action.strip()

        if action.lower() in ("check valid actions", "check_valid_actions"):
            valid = self.domain.get_valid_actions()
            obs = "Valid actions: " + ", ".join(valid)
            return obs, self.domain.get_progress(), False, {"action_is_valid": True}

        if action.lower() == "inventory" or action.lower() == "look":
            obs = self.domain.get_obs()
            return obs, self.domain.get_progress(), False, {"action_is_valid": True}

        valid = self.apply_action(action)
        progress = self.domain.get_progress()
        done = self.domain.check_goal()

        if done:
            self.won = True
            self.domain.done = True
            self.domain.won = True

        obs = self.domain.get_obs()
        if not valid:
            obs = f"Invalid action: '{action}'. Try 'check valid actions' to see available actions.\n" + obs

        return obs, progress, done, {"action_is_valid": valid}

    def apply_action(self, action):
        return self.domain.apply_action(action)

    def reset(self):
        self._setup()
        return self._get_obs(), {}

    @classmethod
    def from_config(cls, cfg):
        return cls(
            game_name=cfg.get("game_name", "gripper"),
            problem_index=cfg.get("problem_index", 0),
            difficulty=cfg.get("difficulty", "easy"),
            label_path=cfg.get("label_path", None),
        )
