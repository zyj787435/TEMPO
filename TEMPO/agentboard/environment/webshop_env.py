"""
webshop_env.py — Offline WebShop text environment for HierInertia evaluation.

Implements a self-contained text-based shopping environment using:
  - BM25 search (rank_bm25) over 1.18M product database
  - Text-based navigation: search → product page → buy
  - Attribute-based reward scoring

Actions:
  search[query]         Search for products
  click[Product N]      Navigate to product N in search results (1-indexed)
  click[option_value]   Select a product option
  click[Buy Now]        Purchase current product
  click[Back]           Return to search results

Reward:
  Price match  : 1.0 if price <= budget, 0 otherwise
  Option score : fraction of required options selected
  Attribute score : fraction of required attributes the product has
  Final reward = price_match * option_score * attribute_score (range 0.0-1.0)
  Success (done=True) only on click[Buy Now]
"""

import json
import os
import re
from pathlib import Path

from agentboard.common.registry import registry

try:
    from rank_bm25 import BM25Okapi
    _BM25_AVAILABLE = True
except ImportError:
    _BM25_AVAILABLE = False

# ── Constants ──────────────────────────────────────────────────────────────

SEARCH_RESULTS_PER_PAGE = 10
MAX_SEARCH_RESULTS = 50   # cap for BM25 top-k

DATA_DIR = Path(__file__).parent.parent / "data" / "data" / "webshop" / "data"
PRODUCTS_PATH = DATA_DIR / "items_shuffle.json"
ATTR_PATH     = DATA_DIR / "items_ins_v2.json"


# ── Offline WebShop environment ────────────────────────────────────────────

@registry.register_environment("webshop")
class Webshop:
    """
    Offline text-based WebShop environment.
    Interface mirrors the HTTP-based version:
      step(session_id, action) -> (observation, reward, done, sub_reward, grounding)
      get_goal() -> str
    """

    def __init__(self, products_path=None, attr_path=None, label_path=None):
        products_path = products_path or str(PRODUCTS_PATH)
        attr_path     = attr_path     or str(ATTR_PATH)

        print("[WebshopEnv] Loading products (this may take ~30s)...", flush=True)
        with open(products_path) as f:
            all_products = json.load(f)

        # Build ASIN → product index
        self._products = {p["asin"]: p for p in all_products if p.get("asin")}
        self._product_list = [p for p in all_products if p.get("asin")]  # ordered list for BM25

        print(f"[WebshopEnv] Loaded {len(self._products):,} products.", flush=True)

        # Load attribute annotations
        self._attr_data = {}
        if os.path.exists(attr_path):
            with open(attr_path) as f:
                self._attr_data = json.load(f)

        # Load test labels (for session mapping)
        self._labels = []
        if label_path and os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    if line.strip():
                        self._labels.append(json.loads(line))

        # Build BM25 index (on product names for speed)
        print("[WebshopEnv] Building BM25 search index...", flush=True)
        self._bm25 = None
        if _BM25_AVAILABLE:
            try:
                def _s(v):
                    """Safely coerce a field value to str (handles list values)."""
                    if isinstance(v, list):
                        return " ".join(str(x) for x in v)
                    return str(v) if v else ""

                corpus = [
                    (_s(p.get("name")) + " " + _s(p.get("small_description")) + " " +
                     _s(p.get("query"))).lower().split()
                    for p in self._product_list
                ]
                self._bm25 = BM25Okapi(corpus)
                print("[WebshopEnv] Search index ready.", flush=True)
            except Exception as e:
                print(f"[WebshopEnv] BM25 build failed ({e}), falling back to keyword search.", flush=True)
                self._bm25 = None
        else:
            print("[WebshopEnv] rank_bm25 not available, using keyword search.", flush=True)

        # Per-session state
        self._sessions: dict = {}
        self.sub_reward = 0.0
        self.reward     = 0.0

    # ── Session helpers ────────────────────────────────────────────────────

    def _init_session(self, session_id: str):
        """Map fixed_N session_id to test task and initialize state."""
        idx = int(session_id.split("_")[-1])
        task = self._labels[idx] if idx < len(self._labels) else None
        if task is None:
            raise ValueError(f"No task for session {session_id}")
        info = task.get("additional_info", {})
        self._sessions[session_id] = {
            "goal": task["goal"],
            "target_asin": info.get("product_id", ""),
            "required_options": info.get("product_options", []),
            "price_upper": float(info.get("price_upper", 1e9)),
            "attributes": info.get("attributes", []),
            "difficulty": task.get("difficulty", "easy"),
            # navigation state
            "page": "search",            # "search" | "product" | "done"
            "search_results": [],        # list of asins from last search
            "current_asin": None,        # currently viewed product
            "selected_options": {},      # {category: value}
            "done": False,
            "last_obs": self._home_obs(task["goal"]),
            "steps": 0,
        }

    def _home_obs(self, goal: str) -> str:
        return (
            f"WebShop\n"
            f"Instruction: {goal}\n\n"
            f"[Search]"
        )

    def get_goal(self) -> str:
        """Return current session goal (call after first step)."""
        return self._current_goal

    # ── Main step interface ────────────────────────────────────────────────

    def step(self, session_id: str, action: str):
        """
        Execute action and return (observation, reward, done, sub_reward, grounding).
        Grounding = True if action is a valid/recognized action.
        """
        if session_id not in self._sessions:
            self._init_session(session_id)
            self.sub_reward = 0.0
            self.reward     = 0.0

        sess = self._sessions[session_id]
        self._current_goal = sess["goal"]
        sess["steps"] += 1

        action = action.strip()
        grounding = True

        if sess["done"]:
            return sess["last_obs"], self.reward, True, self.sub_reward, False

        # ── Parse and dispatch action ──────────────────────────────────────
        obs = ""
        done = False

        if action.lower().startswith("search[") and action.endswith("]"):
            query = action[7:-1].strip()
            obs = self._do_search(session_id, query)

        elif action.lower().startswith("click[") and action.endswith("]"):
            target = action[6:-1].strip()
            obs, done, grounding = self._do_click(session_id, target)

        elif action.lower() == "reset[]":
            # Initial reset — return home page
            obs = sess["last_obs"]
            grounding = True

        else:
            obs = f"Invalid action format. Use search[query] or click[...].\n{sess['last_obs']}"
            grounding = False

        sess["last_obs"] = obs

        # Compute running sub_reward (partial attribute score for viewed product)
        if sess["current_asin"]:
            self.sub_reward = self._score_product(session_id, sess["current_asin"])
        else:
            self.sub_reward = 0.0

        if done:
            self.reward     = self.sub_reward
            sess["done"]    = True

        return obs, self.reward, done, self.sub_reward, grounding

    # ── Action handlers ────────────────────────────────────────────────────

    def _do_search(self, session_id: str, query: str) -> str:
        sess = self._sessions[session_id]
        if not query:
            return "Please enter a search query.\n[Search]"

        results = self._search(query, top_k=MAX_SEARCH_RESULTS)
        sess["search_results"] = [r["asin"] for r in results]
        sess["page"] = "search"
        sess["current_asin"] = None

        lines = [f"Search results for '{query}':\n"]
        for i, r in enumerate(results[:SEARCH_RESULTS_PER_PAGE], 1):
            price_str = r.get("pricing", "N/A")
            name = r.get("name", "Unknown")[:80]
            lines.append(f"{i}. {name}\n   Price: {price_str}")
        if not results:
            lines.append("No results found.")

        # Append clickable options
        lines.append("\n[Back to Search]")
        return "\n".join(lines)

    def _do_click(self, session_id: str, target: str):
        sess = self._sessions[session_id]
        grounding = True

        # ── Buy Now ───────────────────────────────────────────────────────
        if target.lower() == "buy now":
            if sess["current_asin"] is None:
                return "No product selected. Search and click a product first.", False, False
            score = self._score_product(session_id, sess["current_asin"])
            self.sub_reward = score
            prod = self._products.get(sess["current_asin"], {})
            obs = (
                f"You have purchased: {prod.get('name', 'Unknown')}\n"
                f"Final reward: {score:.3f}\n"
                f"{'[Congratulations! Task completed successfully!]' if score >= 1.0 else '[Task completed.]'}"
            )
            return obs, True, True

        # ── Back ──────────────────────────────────────────────────────────
        if target.lower() in ("back", "back to search"):
            obs = self._render_search_results(session_id)
            return obs, False, True

        # ── Click product N (from search results) ────────────────────────
        m = re.match(r"product (\d+)$", target.lower())
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(sess["search_results"]):
                asin = sess["search_results"][idx]
                sess["current_asin"] = asin
                sess["selected_options"] = {}
                sess["page"] = "product"
                return self._render_product_page(session_id), False, True
            else:
                return f"Product {idx + 1} not in results. Valid range: 1-{len(sess['search_results'])}.", False, False

        # ── Click option (on product page) ────────────────────────────────
        if sess["current_asin"]:
            result = self._select_option(session_id, target)
            if result is not None:
                return self._render_product_page(session_id), False, True
            # unrecognized click on product page
            grounding = False
            return (
                f"Option '{target}' not found on this page.\n"
                f"{self._render_product_page(session_id)}"
            ), False, grounding

        # ── Fallback ──────────────────────────────────────────────────────
        grounding = False
        return f"Unrecognized click target: '{target}'.", False, grounding

    # ── Rendering ──────────────────────────────────────────────────────────

    def _render_search_results(self, session_id: str) -> str:
        sess = self._sessions[session_id]
        if not sess["search_results"]:
            return f"No search results. Use search[query].\n[Search]"
        lines = ["Search results:"]
        for i, asin in enumerate(sess["search_results"][:SEARCH_RESULTS_PER_PAGE], 1):
            prod = self._products.get(asin, {})
            lines.append(f"{i}. {prod.get('name', 'Unknown')[:80]}\n   Price: {prod.get('pricing', 'N/A')}")
        lines.append("\n[Back to Search]")
        return "\n".join(lines)

    def _render_product_page(self, session_id: str) -> str:
        sess = self._sessions[session_id]
        asin = sess["current_asin"]
        if not asin:
            return "No product selected."

        prod   = self._products.get(asin, {})
        name   = prod.get("name", "Unknown")
        price  = prod.get("pricing", "N/A")
        desc   = (prod.get("small_description") or prod.get("full_description") or "")[:300]
        cat    = prod.get("product_category", "")

        lines = [
            f"Product: {name}",
            f"Price: {price}",
            f"Category: {cat}",
        ]
        if desc:
            lines.append(f"Description: {desc}")

        # Show options
        opts = prod.get("customization_options")
        if opts and isinstance(opts, dict):
            lines.append("\nOptions:")
            for cat_name, values in opts.items():
                selected = sess["selected_options"].get(cat_name, "")
                if values and isinstance(values, list):
                    val_strs = [
                        f"[{v['value']}]{'*' if v['value'] == selected else ''}"
                        for v in values if isinstance(v, dict) and v.get("value")
                    ]
                    lines.append(f"  {cat_name}: {' '.join(val_strs)}")
                else:
                    lines.append(f"  {cat_name}: (no options)")

        # Show current selections
        if sess["selected_options"]:
            sel_str = ", ".join(f"{k}={v}" for k, v in sess["selected_options"].items())
            lines.append(f"\nSelected: {sel_str}")

        lines.append("\n[Buy Now] [Back to Search]")
        return "\n".join(lines)

    # ── Option selection ───────────────────────────────────────────────────

    def _select_option(self, session_id: str, target: str) -> bool:
        """Try to select target as an option. Return True if found, None otherwise."""
        sess  = self._sessions[session_id]
        prod  = self._products.get(sess["current_asin"], {})
        opts  = prod.get("customization_options")
        if not opts or not isinstance(opts, dict):
            return None

        target_l = target.lower()
        for cat_name, values in opts.items():
            if not values or not isinstance(values, list):
                continue
            for v in values:
                if not isinstance(v, dict):
                    continue
                val = v.get("value", "")
                if val and val.lower() == target_l:
                    sess["selected_options"][cat_name] = val
                    return True
        return None

    # ── BM25 Search ────────────────────────────────────────────────────────

    def _search(self, query: str, top_k: int = MAX_SEARCH_RESULTS):
        if self._bm25 is None:
            # Fallback: simple string match
            q = query.lower()
            results = [p for p in self._product_list if q in (p.get("name", "")).lower()]
            return results[:top_k]

        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [self._product_list[i] for i in indices]

    # ── Reward scoring ─────────────────────────────────────────────────────

    def _score_product(self, session_id: str, asin: str) -> float:
        """
        Score: price_match × option_score × attribute_score
        Range [0.0, 1.0]. Only 1.0 if all criteria satisfied.
        """
        sess = self._sessions[session_id]
        prod = self._products.get(asin, {})
        if not prod:
            return 0.0

        # 1. Price check
        price_str = prod.get("pricing", "")
        price_val = self._parse_price(price_str)
        price_match = 1.0 if price_val <= sess["price_upper"] else 0.0

        # 2. Option score
        required_opts = sess["required_options"]
        if required_opts:
            selected_vals = set(v.lower() for v in sess["selected_options"].values())
            matched_opts  = sum(1 for req in required_opts if req.lower() in selected_vals)
            option_score  = matched_opts / len(required_opts)
        else:
            option_score = 1.0

        # 3. Attribute score
        required_attrs = sess["attributes"]
        if required_attrs:
            prod_attrs = set(a.lower() for a in self._attr_data.get(asin, {}).get("attributes", []))
            # Also check product name and description
            def _sf(v):
                if isinstance(v, list): return " ".join(str(x) for x in v)
                return str(v) if v else ""
            prod_text = (_sf(prod.get("name")) + " " +
                         _sf(prod.get("small_description")) + " " +
                         _sf(prod.get("full_description"))).lower()
            matched_attrs = sum(
                1 for req in required_attrs
                if req.lower() in prod_attrs or req.lower() in prod_text
            )
            attr_score = matched_attrs / len(required_attrs)
        else:
            attr_score = 1.0

        return price_match * option_score * attr_score

    @staticmethod
    def _parse_price(price_str: str) -> float:
        if not price_str:
            return 0.0
        m = re.search(r"[\d,]+\.?\d*", price_str.replace(",", ""))
        if m:
            try:
                return float(m.group().replace(",", ""))
            except ValueError:
                pass
        return 0.0

    # ── Config factory ─────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: dict):
        return cls(
            products_path = cfg.get("products_path", str(PRODUCTS_PATH)),
            attr_path     = cfg.get("attr_path",     str(ATTR_PATH)),
            label_path    = cfg.get("label_path",    None),
        )
