"""
Farming incremental game model and event-driven simulator (fixed bugs)

Fixes implemented:
1) The simulator no longer aborts immediately due to resource checks. It uses configurable
   starting balances (STARTING_COINS, STARTING_AP) so you can reproduce your expected start.
2) Each plot is modelled individually. Every plot can have its own crop+booster and its own
   completion time.
3) The simulation is event-driven ("ticks"). Time advances to the next plot completion,
   harvests are processed individually, and freed plots are replanted according to the
   greedy policy. This matches the requirement that each plot may have different crops/boosters.

Behavior / strategy (greedy):
- While trying to buy the next plot, the simulator targets the next-plot currency (coin or ap)
  and chooses, for each free plot, the best *affordable* crop+booster maximizing net per hour
  in that currency.
- After all plots are owned, it targets AP farming (maximizing AP net/hour).
- If no affordable combo exists for a free plot, that plot stays idle until resources free up.

Usage:
- Edit STARTING_COINS / STARTING_AP at top of the file to match your test scenario.
- Run the script (it prints a step-by-step activity log).

"""

from dataclasses import dataclass
import pandas as pd
from typing import List, Dict, Optional
import sys
import matplotlib.pyplot as plt

# ----------------------
# Configurable parameters
# ----------------------
STARTING_COINS = 10.0   # adjust to reproduce your initial-resource tests
STARTING_AP = 2000.0    # adjust as needed
MAX_PLOTS = 12
MAX_EVENTS = 200000  # safety to prevent infinite loops

# Global AP multiplier for prestige simulation
AP_MULTIPLIER = 1.0

# parameters for simulating prestige. 
# level 0 is the first run. levels 1-n are subsequent runs.
# conditions to trigger next prestige level:
#   - Reach target lifetime AP AND Reach target AP balance
# on prestige, these are reset:
#   All current AP (resets to 15)
#   All current coins (reset to starting amount (10))
#   All owned plots (return to unowned state)
#   All inventory items (seeds, modifiers)
# on prestige, preserved:
#   Total AP earned (lifetime statistic ie `lifetime_ap_gross``)
# prestige reward:
#   AP_MULTIPLIER is multiplied by relevant prestige_ap_multiplier

# no prestige
PRESTIGE_CONFIG_0 = [
    {"level": 0, "target_lifetime_ap": 250_000, "target_ap_balance": 10_000_000, 
     "prestige_ap_multiplier": 1.0, "starting_coins": 10, "starting_ap": 2_000}
]

# 5 prestige levels, each with slightly increasing multipliers and
PRESTIGE_CONFIG_1 = [
    {"level": 0, "target_lifetime_ap": 250_000, "target_ap_balance": 60_000, 
     "prestige_ap_multiplier": 1.0, "starting_coins": 10, "starting_ap": 2_000},

    {"level": 1, "target_lifetime_ap": 250_000, "target_ap_balance": 150_000, 
     "prestige_ap_multiplier": 1.2, "starting_coins": 10, "starting_ap": 15},

    {"level": 2, "target_lifetime_ap": 250_000, "target_ap_balance": 300_000, 
     "prestige_ap_multiplier": 1.4, "starting_coins": 10, "starting_ap": 15},

    {"level": 3, "target_lifetime_ap": 250_000, "target_ap_balance": 500_000, 
     "prestige_ap_multiplier": 1.5, "starting_coins": 10, "starting_ap": 15},

    {"level": 4, "target_lifetime_ap": 250_000, "target_ap_balance": 750_000, 
     "prestige_ap_multiplier": 1.6, "starting_coins": 10, "starting_ap": 15},

    {"level": 5, "target_lifetime_ap": 250_000, "target_ap_balance": 900_000, 
     "prestige_ap_multiplier": 1.8, "starting_coins": 10, "starting_ap": 15},

    {"level": 6, "target_lifetime_ap": 250_000, "target_ap_balance": 1_000_000, 
     "prestige_ap_multiplier": 1.9, "starting_coins": 10, "starting_ap": 15},

    {"level": 7, "target_lifetime_ap": 250_000, "target_ap_balance": 10_000_000, 
     "prestige_ap_multiplier": 2.0, "starting_coins": 10, "starting_ap": 15},

]

# # only reset 3 times
# PRESTIGE_CONFIG_2 = [
#     {"level": 0, "target_lifetime_ap": 250_000, "target_ap_balance": 100_000, "prestige_ap_multiplier": 1.0, "starting_coins": 10, "starting_ap": 2_000},
#     {"level": 1, "target_lifetime_ap": 250_000, "target_ap_balance": 150_000, "prestige_ap_multiplier": 1.1, "starting_coins": 10, "starting_ap": 15},
#     {"level": 2, "target_lifetime_ap": 250_000, "target_ap_balance": 200_000, "prestige_ap_multiplier": 1.2, "starting_coins": 10, "starting_ap": 15},
#     {"level": 3, "target_lifetime_ap": 250_000, "target_ap_balance": 10_000_000, "prestige_ap_multiplier": 1.3, "starting_coins": 10, "starting_ap": 15}
# ]

# # retain more AP with each prestige
# PRESTIGE_CONFIG_3 = [
#     {"level": 0, "target_lifetime_ap": 250_000, "target_ap_balance": 100_000, "prestige_ap_multiplier": 1.0, "starting_coins": 10, "starting_ap": 2_000},
#     {"level": 1, "target_lifetime_ap": 250_000, "target_ap_balance": 150_000, "prestige_ap_multiplier": 1.1, "starting_coins": 10, "starting_ap": 10_000},
#     {"level": 2, "target_lifetime_ap": 250_000, "target_ap_balance": 200_000, "prestige_ap_multiplier": 1.2, "starting_coins": 10, "starting_ap": 15_000},
#     {"level": 3, "target_lifetime_ap": 250_000, "target_ap_balance": 10_000_000, "prestige_ap_multiplier": 1.3, "starting_coins": 10, "starting_ap": 20_000}
# ]


# 5 prestige levels, each one having cumulative *=1.1 ap multiplier and *=1.5 target ap balance
# PRESTIGE_CONFIG_2 = [
#     {"level": 0, "target_lifetime_ap": 250_000, "target_ap_balance": 100_000, "prestige_ap_multiplier": 1.0, "starting_coins": 10, "starting_ap": 2_000},
#     {"level": 1, "target_lifetime_ap": 250_000, "target_ap_balance": 150_000, "prestige_ap_multiplier": 1.1, "starting_coins": 10, "starting_ap": 15},
#     {"level": 2, "target_lifetime_ap": 250_000, "target_ap_balance": 225_000, "prestige_ap_multiplier": 1.21, "starting_coins": 10, "starting_ap": 15},
#     {"level": 3, "target_lifetime_ap": 250_000, "target_ap_balance": 337_500, "prestige_ap_multiplier": 1.331, "starting_coins": 10, "starting_ap": 15},
#     {"level": 4, "target_lifetime_ap": 250_000, "target_ap_balance": 506_250, "prestige_ap_multiplier": 1.4641, "starting_coins": 10, "starting_ap": 15},
#     {"level": 5, "target_lifetime_ap": 250_000, "target_ap_balance": 10_000_000, "prestige_ap_multiplier": 1.61051, "starting_coins": 10, "starting_ap": 15},
# ]
# ----------------------
# Data classes and input
# ----------------------
@dataclass
class PlotSpec:
    plot: int
    cost: float
    currency: str

@dataclass
class CropSpec:
    name: str
    time_s: float
    cost: float
    reward: float
    currency: str
    min_prestige_level: int = 0  # default: available at all levels
    min_prestige_lvl: int = 0

@dataclass
class BoosterSpec:
    name: str
    time_reduction: float
    yield_mult: float
    cost: float
    currency: str
    min_prestige_lvl: int = 0  # Booster can be applied at any prestige level by default

PLOTS: List[PlotSpec] = [
    PlotSpec(1, 0, "coin"), PlotSpec(2, 25, "coin"), PlotSpec(3, 100, "coin"), PlotSpec(4, 500, "coin"),
    PlotSpec(5, 300, "ap"), PlotSpec(6, 1000, "ap"), PlotSpec(7, 2500, "coin"), PlotSpec(8, 2500, "ap"),
    PlotSpec(9, 10000, "coin"), PlotSpec(10, 5000, "ap"), PlotSpec(11, 25000, "coin"), PlotSpec(12, 15000, "ap"),
]

CROPS: List[CropSpec] = [
    CropSpec("wheat", 5, 2, 5, "coin"), CropSpec("lettuce", 30, 8, 15, "coin"), CropSpec("golden AS", 120, 10, 15, "ap"),
    CropSpec("carrot", 180, 25, 50, "coin"), CropSpec("crystal AS", 600, 40, 70, "ap"), CropSpec("tomato", 900, 80, 180, "coin"),
    CropSpec("onion", 3600, 200, 500, "coin"), CropSpec("diamond AS", 3600, 150, 300, "ap"),
    CropSpec("strawberry", 14400, 600, 1500, "coin"), CropSpec("platinum AS", 14400, 500, 1200, "ap"),
    CropSpec("pumpkin", 43200, 1500, 4001, "coin"), CropSpec("royal AS", 43200, 1500, 4001, "ap"),
    # New AP crops with prestige requirements
    CropSpec("Legacy Apple", 60, 5, 15, "ap", min_prestige_level=1),
    CropSpec("Ascendant Apple", 300, 40, 100, "ap", min_prestige_level=2),
    CropSpec("Relic Apple", 2700, 120, 400, "ap", min_prestige_level=3),  # 45m = 2700s
    CropSpec("Ethereal Apple", 7200, 400, 1300, "ap", min_prestige_level=4),  # 2h = 7200s
    CropSpec("Quantum Apple", 36000, 1000, 5000, "ap", min_prestige_level=5),  # 10h = 36000s
    CropSpec("Celestial Apple", 72000, 2500, 8500, "ap", min_prestige_level=6),  # 20h = 72000s
    CropSpec("Apex Apple", 57600, 3000, 12000, "ap", min_prestige_level=7),  # 16h = 57600s
]

BOOSTERS: List[BoosterSpec] = [
    BoosterSpec("(none)", 1.0, 1.0, 0, "none"), BoosterSpec("fertiliser", 0.70, 1.0, 10, "coin"),
    BoosterSpec("silver", 1.0, 1.25, 15, "coin"), BoosterSpec("super", 0.5, 1.0, 25, "ap"),
    BoosterSpec("golden", 1.0, 2.0, 50, "ap"), BoosterSpec("deadly", 0.125, 0.6, 150, "ap"),
    BoosterSpec("quantum", 0.4, 1.5, 175, "ap"),
    # New boosters with prestige requirements
    BoosterSpec("Potion of Gains", 0.6, 1.0, 15, "ap", min_prestige_lvl=1),  # –40% time
    BoosterSpec("Elixir of Degens", 1.0, 1.5, 30, "ap", min_prestige_lvl=2),  # +50% yield
    BoosterSpec("Giga Brew", 0.6, 1.4, 75, "ap", min_prestige_lvl=3),  # –40% time, +40% yield
    BoosterSpec("Wild Growth", 1.25, 3.0, 100, "ap", min_prestige_lvl=4),  # +200% yield, +25% time
    BoosterSpec("Warp-Time Elixir", 0.2, 1.0, 150, "ap", min_prestige_lvl=5),  # –80% time
    BoosterSpec("Titan’s Growth", 1.5, 4.0, 300, "ap", min_prestige_lvl=6),  # +300% yield, +50% time
    BoosterSpec("Apex Potion", 0.25, 2.5, 400, "ap", min_prestige_lvl=7),  # –75% time, +150% yield
]

# ----------------------
# ROI computation
# ----------------------
def build_roi_table(crops: List[CropSpec], boosters: List[BoosterSpec]) -> pd.DataFrame:
    rows = []
    for crop in crops:
        for booster in boosters:
            eff_time_s = crop.time_s * booster.time_reduction
            gross_reward = crop.reward * booster.yield_mult
            coin_cost = crop.cost if crop.currency == "coin" else 0.0
            ap_cost = crop.cost if crop.currency == "ap" else 0.0
            coin_reward = gross_reward if crop.currency == "coin" else 0.0
            ap_reward = gross_reward if crop.currency == "ap" else 0.0
            # For simulation, keep cost columns
            coin_net = coin_reward - coin_cost
            ap_net = ap_reward - ap_cost
            hours = eff_time_s / 3600.0
            coin_net_per_hour = coin_net / hours if hours > 0 else float('-inf')
            ap_net_per_hour = ap_net / hours if hours > 0 else float('-inf')
            ap_gross_per_hour = ap_reward / hours if hours > 0 else 0.0

            rows.append({
                "crop": crop.name,
                "booster": booster.name,
                "coin_cost": coin_cost,
                "ap_cost": ap_cost,
                "coin_reward": coin_reward,
                "ap_reward": ap_reward,
                "coin_net": coin_net,
                "ap_net": ap_net,
                "coin_net_per_hour": coin_net_per_hour,
                "ap_net_per_hour": ap_net_per_hour,
                "ap_gross_per_hour": ap_gross_per_hour,
            })
    return pd.DataFrame(rows)

# ----------------------
# Helper: pick best combo given balances
# ----------------------


def pick_best_combo_for_currency(roi: pd.DataFrame, target_currency: str, coins: float, ap: float, prestige_level: int = 0) -> Optional[Dict]:
    """Pick the best *affordable* combo for the target currency, filtered by prestige level.
    If nothing affordable, return None.
    Preference: maximize net per hour in target currency; fallback to gross reward/hour if needed.
    """
    debug = '--debug' in sys.argv
    def debug_print(msg):
        if debug:
            print(msg)
    if target_currency == "coin":
        metric = "coin_net_per_hour"
        cheapest_coin = min([c.cost for c in CROPS if c.currency == "coin" and getattr(c, 'min_prestige_level', 0) <= prestige_level])
        min_coin_headroom = max(2, 0.3 * cheapest_coin)
    else:
        metric = "ap_net_per_hour"
        cheapest_ap = min([c.cost for c in CROPS if c.currency == "ap" and getattr(c, 'min_prestige_level', 0) <= prestige_level])
        min_ap_headroom = max(2, 0.3 * cheapest_ap)
    debug_print(f"[pick_best_combo_for_currency] Target: {target_currency}, coins={coins}, ap={ap}, prestige_level={prestige_level}")
    # Add headroom checks to affordable combos
    def has_headroom(row):
        if target_currency == "coin":
            coin_left = coins - row["coin_cost"]
            if row["coin_cost"] > 0 and coin_left < min_coin_headroom:
                return False
        else:
            ap_left = ap - row["ap_cost"]
            if row["ap_cost"] > 0 and ap_left < min_ap_headroom:
                return False
        return True
    # Filter by prestige level for crops and boosters
    allowed_crops = [c.name for c in CROPS if getattr(c, 'min_prestige_level', 0) <= prestige_level]
    allowed_boosters = [b.name for b in BOOSTERS if getattr(b, 'min_prestige_lvl', 0) <= prestige_level]
    affordable = roi[(roi["coin_cost"] <= coins) & (roi["ap_cost"] <= ap)].copy()
    affordable = affordable[affordable.apply(has_headroom, axis=1)]
    affordable = affordable[(affordable['crop'].isin(allowed_crops)) & (affordable['booster'].isin(allowed_boosters))]
    debug_print(f"[pick_best_combo_for_currency] Affordable combos (with headroom, prestige): {len(affordable)}")
    if not affordable.empty:
        affordable_sorted = affordable.sort_values(metric, ascending=False)
        debug_print(f"[pick_best_combo_for_currency] Best combo: crop={affordable_sorted.iloc[0]['crop']}, booster={affordable_sorted.iloc[0]['booster']}, {metric}={affordable_sorted.iloc[0][metric]:.2f}")
        return affordable_sorted.iloc[0].to_dict()
    debug_print("[pick_best_combo_for_currency] No affordable combo found.")
    return None

# ----------------------
# Event-driven simulator
# ----------------------

def event_simulator(roi: pd.DataFrame, plotspecs: List[PlotSpec], starting_coins: float, starting_ap: float, lifetime_ap_gross_init: float = 0.0):
    # Switch to enable farming for headroom before buying plots
    # Default FARM_FOR_HEADROOM to True, allow disabling with '--farm-headroom false'
    if '--farm-headroom false' in ' '.join(sys.argv):
        FARM_FOR_HEADROOM = False
    else:
        FARM_FOR_HEADROOM = True

    def min_headroom_for_plot(currency, next_spec):
        if currency == "coin":
            cheapest_coin = min([c.cost for c in CROPS if c.currency == "coin"])
            return max(2, 2 * cheapest_coin, 0.5 * next_spec.cost)
        else:
            cheapest_ap = min([c.cost for c in CROPS if c.currency == "ap"])
            return max(2, 2 * cheapest_ap, 0.5 * next_spec.cost)

    def can_afford_next_plot(balance, plot_cost, min_headroom):
        return balance >= plot_cost + min_headroom

    def buy_next_plot(currency, balance, next_spec):
        min_headroom = min_headroom_for_plot(currency, next_spec)
        if can_afford_next_plot(balance, next_spec.cost, min_headroom):
            return True, min_headroom
        return False, min_headroom

    def farm_for_headroom_action(currency, plots_owned, coins, ap, prestige_level):
        # Farm for headroom by planting best crops for currency
        for pid in range(1, plots_owned+1):
            if plots[pid]["crop"] is None or plots[pid]["busy_until"] <= current_time:
                best_crop_dict = pick_best_combo_for_currency(crop_roi, currency, coins, ap, prestige_level)
                if best_crop_dict is not None:
                    crop = next(c for c in CROPS if c.name == best_crop_dict["crop"])
                    plant_crop(pid, crop)

    crop_usage = {}
    booster_usage = {}
    # For charting: track (elapsed_time, ap_balance)
    ap_time_history = []
    # plot states: dictionary per plot index (1..MAX_PLOTS)
    plots = {}
    for i in range(1, MAX_PLOTS + 1):
        plots[i] = {
            "id": i,
            "owned": True if i == 1 else False,
            "busy_until": 0.0,  # time when current crop completes; 0 means free
            "crop": None,        # CropSpec instance
            "booster": None,     # BoosterSpec instance
            "booster_expiry": 0.0,
        }

    current_time = 0.0
    coins = float(starting_coins)
    ap = float(starting_ap)
    lifetime_ap_gross = lifetime_ap_gross_init
    plots_owned = 1

    debug = '--debug' in sys.argv
    def debug_print(msg):
        if debug:
            print(msg)
    def progress_print(current_time, coins, ap, lifetime_ap_gross, plots_owned, last_progress_time):
        # Print progress every ~1 hour
        if not debug and current_time - last_progress_time >= 3600:
            print(f"[t={current_time/3600:.2f}h] Progress: coins={coins:.1f}, ap={ap:.1f}, lifetime_ap_gross={lifetime_ap_gross:.1f}, plots_owned={plots_owned}")
            return current_time
        return last_progress_time

    print(f"Starting sim: coins={coins}, ap={ap}, plots_owned={plots_owned}")

    def apply_booster(plot_id: int, booster: BoosterSpec) -> bool:
        nonlocal coins, ap, current_time
        # Only apply if expired or not present
        if plots[plot_id]["booster"] is not None and plots[plot_id]["booster_expiry"] > current_time:
            return False
        booster_cost = booster.cost if booster.currency == "coin" else 0.0
        booster_ap_cost = booster.cost if booster.currency == "ap" else 0.0
        if coins < booster_cost or ap < booster_ap_cost:
            return False
        coins -= booster_cost
        ap -= booster_ap_cost
        plots[plot_id]["booster"] = booster
        booster_usage[booster.name] = booster_usage.get(booster.name, 0) + 1
        plots[plot_id]["booster_expiry"] = current_time + 12*3600
        print(f"[t={current_time/3600:.3f}h] Booster {booster.name} applied to plot {plot_id} (expires at t={(plots[plot_id]['booster_expiry']/3600):.3f}h)")
        return True
    def check_booster_expiry(plot_id: int):
        booster = plots[plot_id]["booster"]
        booster_expiry = plots[plot_id]["booster_expiry"]
        if booster is not None and booster_expiry <= current_time:
            print(f"[t={current_time/3600:.3f}h] Booster {booster.name} expired on plot {plot_id}")

    def plant_crop(plot_id: int, crop: CropSpec) -> bool:
        nonlocal coins, ap, current_time
        # Only plant if plot is free
        if plots[plot_id]["crop"] is not None and plots[plot_id]["busy_until"] > current_time:
            return False
        # Headroom checks for planting
        cheapest_coin = min([c.cost for c in CROPS if c.currency == "coin"])
        min_coin_headroom = max(2, 0.3 * cheapest_coin)
        cheapest_ap = min([c.cost for c in CROPS if c.currency == "ap"])
        min_ap_headroom = max(2, 0.3 * cheapest_ap)
        crop_coin_cost = crop.cost if crop.currency == "coin" else 0.0
        crop_ap_cost = crop.cost if crop.currency == "ap" else 0.0
        if coins < crop_coin_cost or ap < crop_ap_cost:
            return False
        if (coins - crop_coin_cost < min_coin_headroom) and crop_coin_cost > 0:
            return False
        if (ap - crop_ap_cost < min_ap_headroom) and crop_ap_cost > 0:
            return False
        # Calculate effective crop duration
        booster = plots[plot_id]["booster"]
        booster_active = booster is not None and plots[plot_id]["booster_expiry"] > current_time
        time_reduction = booster.time_reduction if booster_active else 1.0
        crop_duration = crop.time_s * time_reduction
        coins -= crop_coin_cost
        ap -= crop_ap_cost
        plots[plot_id]["crop"] = crop
        crop_usage[crop.name] = crop_usage.get(crop.name, 0) + 1
        plots[plot_id]["busy_until"] = current_time + crop_duration
        debug_print(f"[t={current_time/3600:.3f}h] Planting {crop.name} on plot {plot_id} — completes at t={(plots[plot_id]['busy_until']/3600):.3f}h; crop cost: {crop_coin_cost}c, {crop_ap_cost}ap")
        return True

    def harvest_plot(plot_id: int):
        nonlocal coins, ap, lifetime_ap_gross, current_time
        crop = plots[plot_id]["crop"]
        if not crop:
            return
        booster = plots[plot_id]["booster"]
        booster_active = booster is not None and plots[plot_id]["booster_expiry"] >= current_time
        yield_mult = booster.yield_mult if booster_active else 1.0
        coin_reward = crop.reward * yield_mult if crop.currency == "coin" else 0.0
        ap_reward = crop.reward * yield_mult if crop.currency == "ap" else 0.0
        ap_reward *= AP_MULTIPLIER
        coins += coin_reward
        ap += ap_reward
        lifetime_ap_gross += ap_reward
        if booster is not None and not booster_active and booster.name != "(none)":
            debug_print(f"[t={current_time/3600:.3f}h] WARNING: Booster {booster.name} not present at harvest on plot {plot_id}")
        debug_print(f"[t={current_time/3600:.3f}h] Harvest on plot {plot_id}: +{coin_reward} coins, +{ap_reward} AP -> balances coins={coins:.1f}, ap={ap:.1f}, lifetime_ap_gross={lifetime_ap_gross:.1f}")
        plots[plot_id]["crop"] = None
        plots[plot_id]["busy_until"] = 0.0

    events_processed = 0
    last_progress_time = 0.0

    # initial planting: attempt to plant any free owned plots
    def plant_all_free_plots():
        nonlocal coins, ap, plots_owned, last_progress_time
        planted_any = False
        # Determine current prestige level
        prestige_level = 0
        if isinstance(plotspecs[-1], dict):
            prestige_level = plotspecs[-1].get('level', 0)
        # Determine current goal: if we still need more plots, target currency is next plot's currency
        target_currency = None
        if plots_owned < MAX_PLOTS:
            next_spec = plotspecs[plots_owned]
            currency = next_spec.currency
            can_buy, min_headroom = buy_next_plot(currency, coins if currency == "coin" else ap, next_spec)
            debug_print(f"[t={current_time/3600:.3f}h] Attempting to buy plot {plots_owned+1}: cost={next_spec.cost}, currency={currency}")
            if can_buy:
                if currency == "coin":
                    coins -= next_spec.cost
                    debug_print(f"[t={current_time/3600:.3f}h] Bought plot {plots_owned} for {next_spec.cost} coins (coins left {coins:.1f})")
                else:
                    ap -= next_spec.cost
                    debug_print(f"[t={current_time/3600:.3f}h] Bought plot {plots_owned} for {next_spec.cost} AP (ap left {ap:.1f})")
                plots_owned += 1
                apply_booster_if_needed(plots_owned)
                plant_all_free_plots()
                last_progress_time = progress_print(current_time, coins, ap, lifetime_ap_gross, plots_owned, last_progress_time)
            # End function after attempting to buy one plot
            return
        else:
            # All plots owned: plant best AP crop on each free plot
            for pid in range(1, plots_owned+1):
                if plots[pid]["crop"] is None or plots[pid]["busy_until"] <= current_time:
                    best_crop_dict = pick_best_combo_for_currency(crop_roi, "ap", coins, ap, prestige_level)
                    if best_crop_dict is not None:
                        crop = next(c for c in CROPS if c.name == best_crop_dict["crop"])
                        plant_crop(pid, crop)
    # Apply boosters to a plot if a new plot is acquired or booster has expired
    def apply_booster_if_needed(plot_id: int):
        nonlocal coins, ap, current_time
        booster = plots[plot_id]["booster"]
        booster_expiry = plots[plot_id]["booster_expiry"]
        # Only apply if expired or not present
        if booster is None or booster_expiry <= current_time:
            # Find best affordable booster, but check coin headroom for coin boosters
            affordable_boosters = []
            for b in BOOSTERS:
                if b.name == "(none)":
                    continue
                if b.currency == "coin":
                    # Check if enough coin remains after purchase for planting a coin crop
                    cheapest_coin_crop = min([c.cost for c in CROPS if c.currency == "coin"])
                    min_coin_headroom = max(2, 0.3 * cheapest_coin_crop)
                    if coins >= b.cost + min_coin_headroom:
                        affordable_boosters.append(b)
                elif b.currency == "ap":
                    cheapest_ap_crop = min([c.cost for c in CROPS if c.currency == "ap"])
                    min_ap_headroom = max(2, 0.3 * cheapest_ap_crop)
                    if ap >= b.cost + min_ap_headroom:
                        affordable_boosters.append(b)
            if affordable_boosters:
                best_booster = max(affordable_boosters, key=lambda b: b.yield_mult)
                apply_booster(plot_id, best_booster)
            else:
                # Not enough coin/ap for any booster, apply '(none)'
                apply_booster(plot_id, next(b for b in BOOSTERS if b.name == "(none)"))

    crop_roi, booster_roi = roi
    # Initial booster application for plot 1
    apply_booster_if_needed(1)
    plant_all_free_plots()
    last_progress_time = progress_print(current_time, coins, ap, lifetime_ap_gross, plots_owned, last_progress_time)

    # Helper to check if we should farm for headroom before buying a plot
    def should_farm_for_headroom(balance, plot_cost, min_headroom):
        debug_print(f"[t={current_time/3600:.3f}h] Checking headroom: balance={balance}, plot_cost={plot_cost}, min_headroom={min_headroom}")
        return (balance - plot_cost) < min_headroom

    # main event loop
    # Get goal values from current prestige config (passed via plotspecs argument)
    prestige_goals = plotspecs[-1] if isinstance(plotspecs[-1], dict) else {}
    target_lifetime_ap = prestige_goals.get('target_lifetime_ap', 0)
    target_ap_balance = prestige_goals.get('target_ap_balance', 0)

    while True:
        if events_processed > MAX_EVENTS:
            print("Reached MAX_EVENTS — aborting to avoid infinite loop")
            break

        # Check termination for this prestige level
        if plots_owned >= MAX_PLOTS and lifetime_ap_gross >= target_lifetime_ap and ap >= target_ap_balance:
            print(f"Goal reached: plots={plots_owned}, lifetime_ap_gross={lifetime_ap_gross:.1f}, ap_balance={ap:.1f}")
            break

        # find next completion time among busy plots
        busy_times = [plots[i]["busy_until"] for i in range(1, plots_owned+1) if plots[i]["crop"] is not None]
        if not busy_times:
            # No busy plots; try to plant again. If we can't plant and cannot buy next plot, we are stuck.
            planted = plant_all_free_plots()
            last_progress_time = progress_print(current_time, coins, ap, lifetime_ap_gross, plots_owned, last_progress_time)
            if planted:
                continue
            # Try to buy next plot if affordable, using helpers and farming switch
            if plots_owned < MAX_PLOTS:
                next_spec = plotspecs[plots_owned]
                currency = next_spec.currency
                can_buy, min_headroom = buy_next_plot(currency, coins if currency == "coin" else ap, next_spec)
                if FARM_FOR_HEADROOM and not can_buy:
                    debug_print(f"[t={current_time/3600:.3f}h] Farming for {currency} headroom before buying plot {plots_owned+1} (need {currency} >= {next_spec.cost + min_headroom}, have {coins if currency == 'coin' else ap:.1f})")
                    prestige_level = 0
                    if isinstance(plotspecs[-1], dict):
                        prestige_level = plotspecs[-1].get('level', 0)
                    farm_for_headroom_action(currency, plots_owned, coins, ap, prestige_level)
                    continue
                if can_buy:
                    if currency == "coin":
                        coins -= next_spec.cost
                        debug_print(f"[t={current_time/3600:.3f}h] Bought plot {plots_owned} for {next_spec.cost} coins (coins left {coins:.1f})")
                    else:
                        ap -= next_spec.cost
                        debug_print(f"[t={current_time/3600:.3f}h] Bought plot {plots_owned} for {next_spec.cost} AP (ap left {ap:.1f})")
                    plots_owned += 1
                    apply_booster_if_needed(plots_owned)
                    plant_all_free_plots()
                    last_progress_time = progress_print(current_time, coins, ap, lifetime_ap_gross, plots_owned, last_progress_time)
                    continue
            # Still cannot do anything -> deadlock
            print("No busy plots and cannot afford any planting or next plot. Simulation halted (deadlock).")
            return

        # advance time to next completion
        next_time = min(busy_times)
        # sanity: next_time should be >= current_time
        if next_time < current_time:
            next_time = current_time
        current_time = next_time

        # process all harvests at this time
        for pid in range(1, plots_owned+1):
            check_booster_expiry(pid)
            if plots[pid]["crop"] is not None and abs(plots[pid]["busy_until"] - current_time) < 1e-6:
                harvest_plot(pid)
                events_processed += 1
                # After harvest, check if booster expired and reapply if needed
                apply_booster_if_needed(pid)
        # Record AP and time for charting
        ap_time_history.append((current_time, ap))

        last_progress_time = progress_print(current_time, coins, ap, lifetime_ap_gross, plots_owned, last_progress_time)

        # After harvests, try to buy as many plots as possible (immediate purchases)
        while plots_owned < MAX_PLOTS:
            next_spec = plotspecs[plots_owned]
            debug_print(f"[t={current_time/3600:.3f}h] Attempting to buy plot {plots_owned+1}: cost={next_spec.cost}, currency={next_spec.currency}")
            if next_spec.currency == "coin":
                cheapest_coin = min([c.cost for c in CROPS if c.currency == "coin"])
                min_headroom = max(2, 2 * cheapest_coin, 0.5 * next_spec.cost)
                if coins >= next_spec.cost + min_headroom:
                    coins -= next_spec.cost
                    plots_owned += 1
                    debug_print(f"[t={current_time/3600:.3f}h] Bought plot {plots_owned} for {next_spec.cost} coins (coins left {coins:.1f})")
                    apply_booster_if_needed(plots_owned)
                    plant_all_free_plots()
                    last_progress_time = progress_print(current_time, coins, ap, lifetime_ap_gross, plots_owned, last_progress_time)
                    continue
            if next_spec.currency == "ap":
                cheapest_ap = min([c.cost for c in CROPS if c.currency == "ap"])
                min_headroom = max(2, 2 * cheapest_ap, 0.5 * next_spec.cost)
                if ap >= next_spec.cost + min_headroom:
                    ap -= next_spec.cost
                    plots_owned += 1
                    debug_print(f"[t={current_time/3600:.3f}h] Bought plot {plots_owned} for {next_spec.cost} AP (ap left {ap:.1f})")
                    apply_booster_if_needed(plots_owned)
                    plant_all_free_plots()
                    last_progress_time = progress_print(current_time, coins, ap, lifetime_ap_gross, plots_owned, last_progress_time)
                    continue
            break

        # After buying, attempt to plant any newly free plots
        plant_all_free_plots()

    # final summary
    print("--- Simulation summary ---")
    print(f"Elapsed time: {current_time/3600:.2f} hours ({current_time/86400:.2f} days)")
    print(f"Final balances: coins={coins:.1f}, ap={ap:.1f}, lifetime_ap_gross={lifetime_ap_gross:.1f}, plots_owned={plots_owned}")
    print("\nCrop usage:")
    for crop, count in sorted(crop_usage.items(), key=lambda x: -x[1]):
        print(f"  {crop}: {count}")
    print("\nBooster usage:")
    for booster, count in sorted(booster_usage.items(), key=lambda x: -x[1]):
        print(f"  {booster}: {count}")
    # Return stats for prestige tracking
    return {
        "lifetime_ap_gross": lifetime_ap_gross,
        "ap": ap,
        "elapsed_time": current_time,
        "ap_time_history": ap_time_history
    }

# ----------------------
# Entrypoint
# ----------------------
def main():
    if '--roi' in sys.argv:
        # Parse prestige level from args, default 0
        prestige_level = 0
        for arg in sys.argv:
            if arg.startswith('--prestige-level='):
                try:
                    prestige_level = int(arg.split('=')[1])
                except Exception:
                    prestige_level = 0
        # Filter crops and boosters by prestige level
        available_crops = [c for c in CROPS if getattr(c, 'min_prestige_level', 0) <= prestige_level]
        available_boosters = [b for b in BOOSTERS if getattr(b, 'min_prestige_lvl', 0) <= prestige_level]
        print(f"Available crops for prestige level {prestige_level}: {[c.name for c in available_crops]}")
        print(f"Available boosters for prestige level {prestige_level}: {[b.name for b in available_boosters]}")
        roi_df = build_roi_table(available_crops, available_boosters)
        roi_print = roi_df[['crop', 'booster', 'coin_net_per_hour', 'ap_net_per_hour', 'coin_reward', 'ap_reward']].copy()
        # Only coin crops in coin profit/hr table
        coin_crops = roi_print[roi_print['coin_reward'] > 0]
        pivot = coin_crops.pivot(index='crop', columns='booster', values='coin_net_per_hour')
        print(f"\nROI Table (coin profit/hr) [Prestige Level {prestige_level}]:")
        print(pivot.round(2).fillna(''))
        # Only AP crops in AP profit/hr table
        ap_crops = roi_print[roi_print['ap_reward'] > 0]
        pivot_ap = ap_crops.pivot(index='crop', columns='booster', values='ap_net_per_hour')
        print(f"\nROI Table (AP profit/hr) [Prestige Level {prestige_level}]:")
        print(pivot_ap.round(2).fillna(''))
        if not coin_crops.empty:
            coin_max_row = coin_crops.loc[coin_crops['coin_net_per_hour'].idxmax()]
            print(f"\nHighest coin ROI: crop = {coin_max_row['crop']}, booster = {coin_max_row['booster']}, coin/hr = {coin_max_row['coin_net_per_hour']:.2f}")
        if not ap_crops.empty:
            ap_max_row = ap_crops.loc[ap_crops['ap_net_per_hour'].idxmax()]
            print(f"Highest AP ROI: crop = {ap_max_row['crop']}, booster = {ap_max_row['booster']}, ap/hr = {ap_max_row['ap_net_per_hour']:.2f}")
        return

    def run_simulation(prestige_config, label, color):
        roi = build_roi_table(CROPS, BOOSTERS)
        total_elapsed_time = 0.0
        prestige_time_markers = []
        ap_time_history_all = []
        lifetime_ap_gross_accum = 0.0
        for idx, prestige in enumerate(prestige_config):
            print(f"\n--- {label} Prestige Level {prestige['level']} ---")
            global AP_MULTIPLIER
            AP_MULTIPLIER = prestige.get("prestige_ap_multiplier", 1.0)
            starting_coins = prestige.get("starting_coins", 10)
            starting_ap = prestige.get("starting_ap", 2000)
            target_lifetime_ap = prestige.get("target_lifetime_ap", 250_000)
            target_ap_balance = prestige.get("target_ap_balance", 0)

            result = event_simulator((roi, None), PLOTS + [prestige], starting_coins, starting_ap, lifetime_ap_gross_init=lifetime_ap_gross_accum)
            if result is not None and isinstance(result, dict):
                lifetime_ap_gross = result.get("lifetime_ap_gross", 0)
                ap_balance = result.get("ap", 0)
                elapsed_time = result.get("elapsed_time", None)
                ap_time_history = result.get("ap_time_history", [])
                # Accumulate lifetime_ap_gross across prestiges
                lifetime_ap_gross_accum = lifetime_ap_gross
                if elapsed_time is not None:
                    total_elapsed_time += elapsed_time
                    prestige_time_markers.append(total_elapsed_time)
                # Offset times by previous elapsed time
                offset = total_elapsed_time - elapsed_time if prestige_time_markers else 0.0
                ap_time_history_all.extend([(t+offset, ap) for t, ap in ap_time_history])
                goal_met = lifetime_ap_gross_accum >= target_lifetime_ap and ap_balance >= target_ap_balance
                if goal_met:
                    print(f"{label} Prestige {prestige['level']} complete: lifetime_ap_gross={lifetime_ap_gross_accum}, ap_balance={ap_balance}")
                    continue
                else:
                    print(f"{label} Prestige {prestige['level']} NOT complete: lifetime_ap_gross={lifetime_ap_gross_accum}, ap_balance={ap_balance}")
                    break
            else:
                continue
        print(f"All {label} prestige levels complete. Total elapsed time: {total_elapsed_time/3600:.2f} hours ({total_elapsed_time/86400:.2f} days)")
        return ap_time_history_all, prestige_time_markers

    chart_enabled = '--chart' in sys.argv
    # Run both simulations
    ap_time_0, markers_0 = run_simulation(PRESTIGE_CONFIG_0, "Config0", 'b')
    ap_time_1, markers_1 = run_simulation(PRESTIGE_CONFIG_1, "Config1", 'g')
    # ap_time_2, markers_2 = run_simulation(PRESTIGE_CONFIG_2, "Config2", 'r')
    # ap_time_3, markers_3 = run_simulation(PRESTIGE_CONFIG_3, "Config3", 'c')

    # Chart output if requested
    if chart_enabled:
        plt.figure(figsize=(10,6))
        # Plot Config0
        if ap_time_0:
            times0 = [t/3600.0/24.0 for t, ap in ap_time_0]
            aps0 = [ap for t, ap in ap_time_0]
            plt.plot(times0, aps0, label='PRESTIGE_CONFIG_0', color='b')
            for marker in markers_0:
                plt.axvline(x=marker/3600.0/24.0, color='b', linestyle='--', alpha=0.5)
        # Plot Config1
        if ap_time_1:
            times1 = [t/3600.0/24.0 for t, ap in ap_time_1]
            aps1 = [ap for t, ap in ap_time_1]
            plt.plot(times1, aps1, label='PRESTIGE_CONFIG_1', color='g')
            for marker in markers_1:
                plt.axvline(x=marker/3600.0/24.0, color='g', linestyle='--', alpha=0.5)
        # # Plot Config2
        # if ap_time_2:
        #     times2 = [t/3600.0/24.0 for t, ap in ap_time_2]
        #     aps2 = [ap for t, ap in ap_time_2]
        #     plt.plot(times2, aps2, label='PRESTIGE_CONFIG_2', color='r')
        #     for marker in markers_2:
        #         plt.axvline(x=marker/3600.0/24.0, color='r', linestyle='--', alpha=0.5)
        # # Plot Config3
        # if ap_time_3:
        #     times3 = [t/3600.0/24.0 for t, ap in ap_time_3]
        #     aps3 = [ap for t, ap in ap_time_3]
        #     plt.plot(times3, aps3, label='PRESTIGE_CONFIG_3', color='c')
        #     for marker in markers_3:
        #         plt.axvline(x=marker/3600.0/24.0, color='c', linestyle='--', alpha=0.5)
        plt.xlabel('Elapsed Time (days)')
        plt.ylabel('AP Balance')
        plt.title('AP Balance vs Elapsed Time (Multiple Prestige Configs)')
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
