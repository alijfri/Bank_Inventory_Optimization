from ortools.linear_solver import pywraplp
from math import inf
import pandas as pd

from inputs import get_trip_config, CONFIG, start_date, finish_date, costs, target_freq, num_notes_in_bag, rdp_to_be_solved
from weekly_opt.weekly_scenario_RHC import (
    prediction_unfit_weighted_average_weekly,
    attach_weekly_unfit_predictions_to_network,
    prediction_weighted_average_weekly,
    attach_weekly_predictions_to_network,
    generate_lhs_scenarios,
)


class BanknoteInventoryModel:
    def __init__(
        self,
        network,
        order_policy,
        initial_inventory,
        opt_start_date_block,
        opt_end_date_block,
        num_opt_weeks,
        initial_unfit=None,
        solver_name="SCIP",
    ):
        self.solver = pywraplp.Solver.CreateSolver(solver_name)
        if self.solver is None:
            raise RuntimeError(f"Could not create OR-Tools solver '{solver_name}'")

        self.network = network
        self.costs = costs
        self.order_policy = order_policy  # 'separate_ss' or 'joint_ss'

        # === match Pyomo knobs ===
        self.n_scen = 10
        self.band_half_width_pct = 0.10
        self.scale_factor = 1000

        # initial states (already in "notes"/value units like Pyomo inputs)
        self.initial_inventory = initial_inventory or {}
        self.initial_unfit = initial_unfit or {}

        self.opt_start_date_block = opt_start_date_block
        self.opt_end_date_block = opt_end_date_block
        self.num_opt_weeks = num_opt_weeks

        # scale num_notes_in_bag like Pyomo
        self.num_notes_in_bag = num_notes_in_bag / 1000

        self._build_sets()
        self._build_parameters()
        self._build_variables()
        self._build_constraints()
        self._build_objective()

    # ---------- sets ----------
    def _build_sets(self):
        # identical spirit to your Pyomo _build_sets
        self.I = [rdp_to_be_solved]
        self.B = [5, 10, 20, 50, 100]
        self.NT = ["FIT", "NEW", "UNFIT"]
        self.N_active = ["FIT", "NEW"]
        self.T = list(range(0, self.num_opt_weeks))

        # review days (weekly model: you used "return [t for t in m.T]" => effectively all periods)
        self.review_days = {i: CONFIG[i]["rep_plan_day"] for i in self.I}

        # In your Pyomo, review_time_rule returns [t for t in m.T] -> all times are review times
        self.T_review = {i: [t for t in self.T] for i in self.I}
        self.IT_review = [(i, t) for i in self.I for t in self.T_review[i]]

        self.Omega = list(range(self.n_scen))

        # arcs existed in Pyomo but not used in constraints you pasted; keep placeholder
        self.Arcs = []

    # ---------- parameters ----------
    def _build_parameters(self):
        net = self.network

        # --- Demand scenarios D[b,i,t,n,omega] ---
        preds = prediction_weighted_average_weekly(net, self.opt_start_date_block, self.opt_end_date_block)
        attach_weekly_predictions_to_network(net, preds)

        self.D = {}  # (b,i,t,n,omega) -> scaled float
        for i in self.I:
            rdp = net.rdps[i]
            for b in self.B:
                for n in self.N_active:
                    pred_full = rdp.pred_series(b, n)
                    scen = generate_lhs_scenarios(
                        pred_full,
                        n_scen=self.n_scen,
                        band_half_width_pct=self.band_half_width_pct,
                        seed=42,
                    )
                    T_use = min(len(self.T), scen.shape[1])
                    for t in range(T_use):
                        for w in self.Omega:
                            self.D[(b, i, t, n, w)] = int(scen[w, t] / self.scale_factor)

        # default 0 if missing
        def D_get(b, i, t, n, w):
            return self.D.get((b, i, t, n, w), 0.0)

        self.D_get = D_get

        # --- Unfit scenarios unfit_value[i,t,omega] ---
        preds_unfit = prediction_unfit_weighted_average_weekly(net, self.opt_start_date_block, self.opt_end_date_block)
        attach_weekly_unfit_predictions_to_network(net, preds_unfit)

        self.unfit_value = {}  # (i,t,omega)
        for i in self.I:
            rdp = net.rdps[i]
            unfit_pred = rdp.unfit_pred_series()
            scn_unfit = generate_lhs_scenarios(
                unfit_pred,
                n_scen=self.n_scen,
                band_half_width_pct=self.band_half_width_pct,
                seed=42,
            )
            T_use = min(len(self.T), scn_unfit.shape[1])
            for t in range(T_use):
                for w in self.Omega:
                    self.unfit_value[(i, t, w)] = int(scn_unfit[w, t] / self.scale_factor)

        def unfit_get(i, t, w):
            return self.unfit_value.get((i, t, w), 0.0)

        self.unfit_get = unfit_get

        # capacity + bigMs + lead time
        self.Cap = {i: int(net.rdps[i].get_capacity() / self.scale_factor) for i in self.I}
        self.big_M = int(80000000 / self.scale_factor)
        self.bigM_2 = int(40000000 / self.scale_factor)
        self.bigM_3 = int(3000000 / self.scale_factor)

        # weekly model lead time fixed to 1 in your Pyomo
        self.L = {i: 1 for i in self.I}

        self.max_shipment = {i: int(get_trip_config(i)["max_value"] / self.scale_factor) for i in self.I}
        self.max_bags = {i: int(get_trip_config(i)["max_bags_per_trip"]) for i in self.I}
        self.unfit_removal = {i: int(CONFIG[i]["weekly_unfit_removal"]) for i in self.I}

        # InitUnfit
        self.InitUnfit = {}
        for i in self.I:
            self.InitUnfit[i] = int(self.initial_unfit.get(i, 0.0) / self.scale_factor)

        # InitInv[b,i,n]
        self.InitInv = {}
        for (b, i, n), val in self.initial_inventory.items():
            self.InitInv[(b, i, n)] = int(val / self.scale_factor)
        # If not provided for some keys, default to 0 like your current Pyomo behavior
        for i in self.I:
            for b in self.B:
                for n in self.N_active:
                    self.InitInv.setdefault((b, i, n), 0.0)

        # g_lo/g_hi (you set both to 0.5*Cap currently)
        self.g_lo = {}
        self.g_hi = {}
        for b in self.B:
            for i in self.I:
                for n in self.N_active:
                    S_I_min = 0.5 * self.Cap[i]
                    S_I_max = 0.5 * self.Cap[i]
                    self.g_lo[(b, i, n)] = S_I_min
                    self.g_hi[(b, i, n)] = S_I_max

    # ---------- variables ----------
    def _build_variables(self):
        s = self.solver

        # Inventory and transitions
        self.I_bn = {}      # (b,i,t,n,omega)
        self.I_unfit = {}   # (i,t,omega)
        self.er = {}        # (b,i,t,n,omega)
        self.x_new_to_fit = {}  # (b,i,t,omega)

        for b in self.B:
            for i in self.I:
                for t in self.T:
                    for w in self.Omega:
                        self.x_new_to_fit[(b, i, t, w)] = s.NumVar(0.0, inf, f"x_new_to_fit[{b},{i},{t},{w}]")
                        for n in self.N_active:
                            self.I_bn[(b, i, t, n, w)] = s.NumVar(0.0, inf, f"I_bn[{b},{i},{t},{n},{w}]")
                            self.er[(b, i, t, n, w)] = s.NumVar(0.0, inf, f"er[{b},{i},{t},{n},{w}]")
                            

        for i in self.I:
            for t in self.T:
                for w in self.Omega:
                    self.I_unfit[(i, t, w)] = s.NumVar(0.0, inf, f"I_unfit[{i},{t},{w}]")

        # thresholds s,S
        self.s_var = {}  # (b,i,n)
        self.S_var = {}  # (b,i,n)
        for b in self.B:
            for i in self.I:
                for n in self.N_active:
                    # match Pyomo: s lower bound is network.get_lower_bound(b,n)/scale_factor
                    s_lo = float(self.network.rdps[i].get_lower_bound(b, n)) / self.scale_factor
                    self.s_var[(b, i, n)] = s.NumVar(s_lo, inf, f"s[{b},{i},{n}]")
                    # match Pyomo S_bounds: lb=0 ub=Cap/b
                    self.S_var[(b, i, n)] = s.NumVar(0.0, self.Cap[i] / b, f"S[{b},{i},{n}]")

        # replenishment q only defined on IT_review in Pyomo: q[b, (i,t), n, omega]
        self.q = {}  # (b,i,t,n,omega) but only for (i,t) in IT_review
        self.h = {}  # (b,i,t,n,omega) only for IT_review
        for (i, t) in self.IT_review:
            for b in self.B:
                for n in self.N_active:
                    for w in self.Omega:
                        self.q[(b, i, t, n, w)] = s.NumVar(0.0, inf, f"q[{b},{i},{t},{n},{w}]")
                        self.h[(b, i, t, n, w)] = s.BoolVar(f"h[{b},{i},{t},{n},{w}]")

        # callbacks
        self.w = {}   # (b,i,t,omega)
        self.U = {}   # (i,t,omega)
        self.O = {}   # (i,t,omega)
        self.round_trip = {}  # (i,t,omega)
        for i in self.I:
            for t in self.T:
                for w_ in self.Omega:
                    self.O[(i, t, w_)] = s.BoolVar(f"O[{i},{t},{w_}]")
                    self.round_trip[(i, t, w_)] = s.BoolVar(f"round_trip[{i},{t},{w_}]")
                    self.U[(i, t, w_)] = s.NumVar(0.0, inf, f"U[{i},{t},{w_}]")
                    for b in self.B:
                        self.w[(b, i, t, w_)] = s.NumVar(0.0, inf, f"w[{b},{i},{t},{w_}]")

        # joint trigger
        self.joint_h = {}  # (i,t,omega) only for IT_review
        for (i, t) in self.IT_review:
            for w_ in self.Omega:
                self.joint_h[(i, t, w_)] = s.BoolVar(f"joint_h[{i},{t},{w_}]")

        # y_fit only for joint_ss
        self.y_fit = {}
        if self.order_policy == "joint_ss":
            for (i, t) in self.IT_review:
                for b in self.B:
                    for w_ in self.Omega:
                        self.y_fit[(b, i, t, w_)] = s.BoolVar(f"y_fit[{b},{i},{t},{w_}]")

    # ---------- constraints ----------
    def _build_constraints(self):
        s = self.solver

        # helpers
        def q_var(b, i, t, n, w):
            # only exists for (i,t) in IT_review
            return self.q[(b, i, t, n, w)]

        def is_review(i, t):
            return (i, t) in set(self.IT_review)

        # --- NEW balance ---
        for b in self.B:
            for i in self.I:
                L = int(self.L[i])
                for t in self.T:
                    for w in self.Omega:
                        incoming = 0.0
                        tau = t - L
                        if tau >= 0 and is_review(i, tau):
                            incoming = q_var(b, i, tau, "NEW", w)

                        if t == 0:
                            s.Add(
                                self.I_bn[(b, i, 0, "NEW", w)]
                                == self.InitInv[(b, i, "NEW")]
                                - self.D_get(b, i, 0, "NEW", w)
                                + self.er[(b, i, 0, "NEW", w)]
                                - self.x_new_to_fit[(b, i, 0, w)]
                            )
                        else:
                            s.Add(
                                self.I_bn[(b, i, t, "NEW", w)]
                                == self.I_bn[(b, i, t - 1, "NEW", w)]
                                + incoming
                                + self.er[(b, i, t, "NEW", w)]
                                - self.D_get(b, i, t, "NEW", w)
                                - self.x_new_to_fit[(b, i, t, w)]
                            )

        # --- FIT balance ---
        for b in self.B:
            for i in self.I:
                L = int(self.L[i])
                for t in self.T:
                    for w in self.Omega:
                        incoming = 0.0
                        tau = t - L
                        if tau >= 0 and is_review(i, tau):
                            incoming = q_var(b, i, tau, "FIT", w)

                        if t == 0:
                            s.Add(
                                self.I_bn[(b, i, 0, "FIT", w)]
                                == self.InitInv[(b, i, "FIT")]
                                + self.er[(b, i, 0, "FIT", w)]
                                + self.x_new_to_fit[(b, i, 0, w)]
                                - self.D_get(b, i, 0, "FIT", w)
                                - self.w[(b, i, 0, w)]
                            )
                        else:
                            s.Add(
                                self.I_bn[(b, i, t, "FIT", w)]
                                == self.I_bn[(b, i, t - 1, "FIT", w)]
                                + incoming
                                + self.er[(b, i, t, "FIT", w)]
                                + self.x_new_to_fit[(b, i, t, w)]
                                - self.D_get(b, i, t, "FIT", w)
                                - self.w[(b, i, t, w)]
                            )

        # --- UNFIT balance (with InitUnfit) ---
        for i in self.I:
            for t in self.T:
                for w in self.Omega:
                    if t == 0:
                        s.Add(
                            self.I_unfit[(i, 0, w)]
                            == self.InitUnfit[i] + self.unfit_get(i, 0, w)
                        )
                    else:
                        s.Add(
                            self.I_unfit[(i, t, w)]
                            == self.I_unfit[(i, t - 1, w)]
                            + self.unfit_get(i, t, w)
                            - self.U[(i, t - 1, w)]
                        )

        # --- s + 1 <= S ---
        for b in self.B:
            for i in self.I:
                for n in self.N_active:
                    s.Add(self.s_var[(b, i, n)] + 1 <= self.S_var[(b, i, n)])

        # --- Overcap trigger at 0.9 Cap + removal rules (exactly like Pyomo) ---
        for i in self.I:
            for t in self.T:
                for w in self.Omega:
                    VOH = sum(self.I_bn[(b, i, t, n, w)] * b for b in self.B for n in self.N_active)

                    # OvercapTrigger: I_unfit + VOH - 0.9Cap <= bigM_2 * O
                    s.Add(self.I_unfit[(i, t, w)] + VOH - 0.9 * self.Cap[i] <= self.bigM_2 * self.O[(i, t, w)])

                    # RemoveExcessWhenNeeded:
                    removal = self.U[(i, t, w)] + sum(self.w[(b, i, t, w)] * b for b in self.B)
                    s.Add(removal >= (VOH + self.I_unfit[(i, t, w)] - 0.9 * self.Cap[i]) - self.bigM_2 * (1 - self.O[(i, t, w)]))

                    # OvercapLogic (U equals I_unfit when O=1 else 0-ish)
                    s.Add(self.U[(i, t, w)] <= self.bigM_2 * self.O[(i, t, w)])
                    s.Add(self.U[(i, t, w)] <= self.I_unfit[(i, t, w)])
                    s.Add(self.U[(i, t, w)] >= self.I_unfit[(i, t, w)] - self.bigM_2 * (1 - self.O[(i, t, w)]))

                    # Callback capacity (value)
                    s.Add(
                        self.U[(i, t, w)] + sum(self.w[(b, i, t, w)] * b for b in self.B)
                        <= self.max_shipment[i] * self.O[(i, t, w)]
                    )
                    # Callback capacity (notes/bags): U/20 + sum(w) <= num_notes_in_bag*max_bags * O
                    s.Add(
                        self.U[(i, t, w)] / 20.0 + sum(self.w[(b, i, t, w)] for b in self.B)
                        <= (self.num_notes_in_bag * self.max_bags[i]) * self.O[(i, t, w)]
                    )

        # --- Sliding window callbacks (your Pyomo uses weekly_unfit_removal length) ---
        for i in self.I:
            W = int(self.unfit_removal[i])
            for t in self.T:
                if t > max(self.T) - W + 1:
                    continue
                for w in self.Omega:
                    s.Add(sum(self.O[(i, tau, w)] for tau in range(t, t + W)) >= 1)

        # --- Round trip logic (same as Pyomo) ---
        for i in self.I:
            L = int(self.L[i])
            for t in self.T:
                for w in self.Omega:
                    tau = t - L
                    if tau < 0:
                        s.Add(self.round_trip[(i, t, w)] == 0)
                    elif not is_review(i, tau):
                        s.Add(self.round_trip[(i, t, w)] == 0)
                    else:
                        s.Add(self.joint_h[(i, tau, w)] >= self.round_trip[(i, t, w)])
                    s.Add(self.O[(i, t, w)] >= self.round_trip[(i, t, w)])

        # --- JointTriggerUB: joint_h <= sum h ---
        for (i, t) in self.IT_review:
            for w in self.Omega:
                s.Add(
                    self.joint_h[(i, t, w)]
                    <= sum(self.h[(b, i, t, n, w)] for b in self.B for n in self.N_active)
                )

        # --- separate_ss vs joint_ss ordering constraints ---
        if self.order_policy == "separate_ss":
            # trigger_upper/lower on review days only (we only have h,q for IT_review anyway)
            for (i, t) in self.IT_review:
                for w in self.Omega:
                    for b in self.B:
                        for n in self.N_active:
                            # I <= s + M(1-h)
                            s.Add(self.I_bn[(b, i, t, n, w)] <= self.s_var[(b, i, n)] + self.bigM_3 * (1 - self.h[(b, i, t, n, w)]))
                            # I + M*h >= s+1
                            s.Add(self.I_bn[(b, i, t, n, w)] + self.bigM_3 * self.h[(b, i, t, n, w)] >= self.s_var[(b, i, n)] + 1)

                            # JR bounds for NEW
                            if n == "NEW":
                                s.Add(self.q[(b, i, t, "NEW", w)] >= (self.S_var[(b, i, "NEW")] - self.I_bn[(b, i, t, "NEW", w)]) - self.bigM_3 * (1 - self.h[(b, i, t, "NEW", w)]))
                                s.Add(self.q[(b, i, t, "NEW", w)] <= (self.S_var[(b, i, "NEW")] - self.I_bn[(b, i, t, "NEW", w)]) + self.bigM_3 * (1 - self.h[(b, i, t, "NEW", w)]))
                            # JR bounds for FIT
                            if n == "FIT":
                                s.Add(self.q[(b, i, t, "FIT", w)] >= (self.S_var[(b, i, "FIT")] - self.I_bn[(b, i, t, "FIT", w)]) - self.bigM_3 * (1 - self.h[(b, i, t, "FIT", w)]))
                                s.Add(self.q[(b, i, t, "FIT", w)] <= (self.S_var[(b, i, "FIT")] - self.I_bn[(b, i, t, "FIT", w)]) + self.bigM_3 * (1 - self.h[(b, i, t, "FIT", w)]))

                            # Gate q <= bigM_3 * h
                            s.Add(self.q[(b, i, t, n, w)] <= self.bigM_3 * self.h[(b, i, t, n, w)])

        elif self.order_policy == "joint_ss":
            for (i, t) in self.IT_review:
                for w in self.Omega:
                    # trigger_upper/lower still uses per-item h in your Pyomo joint_ss block
                    for b in self.B:
                        for n in self.N_active:
                            s.Add(self.I_bn[(b, i, t, n, w)] <= self.s_var[(b, i, n)] + self.bigM_3 * (1 - self.h[(b, i, t, n, w)]))
                            s.Add(self.I_bn[(b, i, t, n, w)] + self.bigM_3 * self.h[(b, i, t, n, w)] >= self.s_var[(b, i, n)] + 1)

                    # JR for NEW uses joint_h
                    for b in self.B:
                        s.Add(self.q[(b, i, t, "NEW", w)] >= (self.S_var[(b, i, "NEW")] - self.I_bn[(b, i, t, "NEW", w)]) - self.bigM_3 * (1 - self.joint_h[(i, t, w)]))
                        s.Add(self.q[(b, i, t, "NEW", w)] <= (self.S_var[(b, i, "NEW")] - self.I_bn[(b, i, t, "NEW", w)]) + self.bigM_3 * (1 - self.joint_h[(i, t, w)]))

                    # FIT JR uses y_fit and joint_h like your Pyomo: - bigM_3*(1 - y + 1 - joint_h)
                    for b in self.B:
                        y = self.y_fit[(b, i, t, w)]
                        jh = self.joint_h[(i, t, w)]
                        s.Add(self.q[(b, i, t, "FIT", w)] >= (self.S_var[(b, i, "FIT")] - self.I_bn[(b, i, t, "FIT", w)]) - self.bigM_3 * ((1 - y) + (1 - jh)))
                        s.Add(self.q[(b, i, t, "FIT", w)] <= (self.S_var[(b, i, "FIT")] - self.I_bn[(b, i, t, "FIT", w)]) + self.bigM_3 * ((1 - y) + (1 - jh)))

                        # s_I_upper_rule: S - I <= bigM_3 * y_fit
                        s.Add(self.S_var[(b, i, "FIT")] - self.I_bn[(b, i, t, "FIT", w)] <= self.bigM_3 * y)

                        # Gate mode fit: q_fit <= bigM_3 * y_fit
                        s.Add(self.q[(b, i, t, "FIT", w)] <= self.bigM_3 * y)

                    # (Your Pyomo had JR_Gate_Mode commented out for joint case; keep it commented here too)


        # --- JointTrigger: joint_h >= h[b,it,n] ---
        for (i, t) in self.IT_review:
            for w in self.Omega:
                for b in self.B:
                    for n in self.N_active:
                        s.Add(self.joint_h[(i, t, w)] >= self.h[(b, i, t, n, w)])

        # --- Capacity on Big-S: sum S*b <= 0.9 Cap (your Pyomo uses 0.9*Cap) ---
        for i in self.I:
            s.Add(sum(self.S_var[(b, i, n)] * b for b in self.B for n in self.N_active) <= int(0.9 * self.Cap[i]))

        # --- Shipment caps on review days, gated by joint_h ---
        for (i, t) in self.IT_review:
            for w in self.Omega:
                s.Add(
                    sum(self.q[(b, i, t, n, w)] * b for b in self.B for n in self.N_active)
                    <= self.max_shipment[i] * self.joint_h[(i, t, w)]
                )
                s.Add(
                    sum(self.q[(b, i, t, n, w)] for b in self.B for n in self.N_active)
                    <= (self.num_notes_in_bag * self.max_bags[i]) * self.joint_h[(i, t, w)]
                )

    # ---------- objective ----------
    def _build_objective(self):
        s = self.solver
        c = self.costs

        # replicate your Pyomo objective exactly
        ER_cost = sum(self.er[(b, i, t, n, w)] for b in self.B for i in self.I for t in self.T for n in self.N_active for w in self.Omega)
        repl_var = sum(self.q[(b, i, t, n, w)] for (i, t) in self.IT_review for b in self.B for n in self.N_active for w in self.Omega)
        overcap_fixed = sum(self.O[(i, t, w)] for i in self.I for t in self.T for w in self.Omega)
        overcap_var = sum(self.w[(b, i, t, w)] for b in self.B for i in self.I for t in self.T for w in self.Omega)
        convert_pen = sum(self.x_new_to_fit[(b, i, t, w)] for b in self.B for i in self.I for t in self.T for w in self.Omega)
        discount = sum(self.round_trip[(i, t, w)] for i in self.I for t in self.T for w in self.Omega)
        repl_fixed = sum(self.joint_h[(i, t, w)] for (i, t) in self.IT_review for w in self.Omega)
        repl_per_item = sum(self.h[(b, i, t, n, w)] for (i, t) in self.IT_review for b in self.B for n in self.N_active for w in self.Omega)
        holdinc_cost = sum(self.I_bn[(b, i, t, n, w)] for b in self.B for i in self.I for t in self.T for n in self.N_active for w in self.Omega)

        total_cost = (1.0 / self.n_scen) * (
            c.c_er * ER_cost
            + repl_per_item * c.c_h * 0.33
            + c.c_h * repl_fixed
            + c.c_q * repl_var
            + c.c_O * overcap_fixed
            + c.c_w * overcap_var
            + c.c_x * convert_pen
            - 0.1 * c.c_h * discount
        )

        s.Minimize(total_cost)

    # ---------- solve ----------
    def solve(self, mip_gap=0.01, time_limit_sec=500, verbose=True):
        # SCIP params via SetSolverSpecificParametersAsString
        # Note: exact parameter names differ by backend; SCIP supports limits/time and gap
        params = []
        if time_limit_sec is not None:
            params.append(f"limits/time = {float(time_limit_sec)}")
        if mip_gap is not None:
            params.append(f"limits/gap = {float(mip_gap)}")
        if params:
            self.solver.SetSolverSpecificParametersAsString("\n".join(params))

        if verbose:
            self.solver.EnableOutput()     # <- no argument
        else:
            self.solver.SuppressOutput()   # <- use this to silence

        return self.solver.Solve()

    def report_results(self, tol=1e-6, float_fmt="{:.0f}", scenario=0):
        """
        OR-Tools version of your Pyomo report_results().
        Prints ONLY one scenario (default scenario=0), like your Pyomo code.
        """
        s = self.solver
        I, B, N_active, T = self.I, self.B, self.N_active, self.T
        w0 = int(scenario)

        def val(x):
            # OR-Tools: NumVar/BoolVar -> solution_value()
            return float(x.solution_value())

        def nz_series(vals):
            """Return only (t:value) pairs where value != 0 (above tol)."""
            return ", ".join(
                [f"t={t}:{float_fmt.format(float(v))}" for t, v in enumerate(vals) if v and abs(v) > tol]
            ) or "all 0"

        print("\n=== OBJECTIVE VALUE ===")
        print(f"Objective: {s.Objective().Value()}")
        print(f"Number of scenarios: {self.n_scen}")
        print(f"Printed scenario: {w0}")

        # ---------------- INVENTORY ----------------
        print("\n=== INVENTORY LEVELS (FIT/NEW) ===")
        for i in I:
            for b in B:
                for n in N_active:
                    inv_series = [val(self.I_bn[(b, i, t, n, w0)]) for t in T]
                    full = ", ".join([f"t={t}:{float_fmt.format(v)}" for t, v in enumerate(inv_series)])
                    print(f"RDP={i}, Denom={b}, NoteType={n}: {full}")

        print("\n=== UNFIT INVENTORY ===")
        for i in I:
            inv_series = [val(self.I_unfit[(i, t, w0)]) for t in T]
            print(f"RDP={i}: {nz_series(inv_series)}")

        # ---------------- DECISIONS ----------------
        print("\n=== REPLENISHMENT DECISIONS (q) ===")
        for i in I:
            for b in B:
                for n in N_active:
                    q_series = []
                    for t in T:
                        # q exists only on IT_review
                        if (i, t) in self.IT_review:
                            q_val = val(self.q[(b, i, t, n, w0)])
                        else:
                            q_val = 0.0
                        q_series.append(q_val)

                    nz = nz_series(q_series)
                    if nz != "all 0":
                        print(f"RDP={i}, Denom={b}, NoteType={n}: {nz}")

        print("\n=== EMERGENCY REPLENISHMENT (er) ===")
        for i in I:
            for b in B:
                for n in N_active:
                    er_series = [val(self.er[(b, i, t, n, w0)]) for t in T]
                    nz = nz_series(er_series)
                    if nz != "all 0":
                        print(f"RDP={i}, Denom={b}, NoteType={n}: {nz}")

        print("\n=== NEW → FIT CONVERSION (x_new_to_fit) ===")
        for i in I:
            for b in B:
                x_series = [val(self.x_new_to_fit[(b, i, t, w0)]) for t in T]
                nz = nz_series(x_series)
                if nz != "all 0":
                    print(f"RDP={i}, Denom={b}: {nz}")

        # (You had alpha printing, but alpha is not defined in the OR-Tools model you shared)
        if hasattr(self, "alpha"):
            print("\n=== overcap threshold (alpha) ===")
            for i in I:
                print(f"RDP={i}, alpha: {val(self.alpha[i])}")

        # ---------------- CALLBACKS ----------------
        print("\n=== CALLBACK DECISIONS ===")
        for i in I:
            u_series = [val(self.U[(i, t, w0)]) for t in T]
            nz_u = nz_series(u_series)
            if nz_u != "all 0":
                print(f"RDP={i}, U (unfit value): {nz_u}")

            for b in B:
                w_series = [val(self.w[(b, i, t, w0)]) for t in T]
                nz_w = nz_series(w_series)
                if nz_w != "all 0":
                    print(f"RDP={i}, Denom={b}, w (callback notes): {nz_w}")

            for t in T:
                if val(self.U[(i, t, w0)]) > tol:
                    voh = sum(val(self.I_bn[(b, i, t, n, w0)]) * b for b in B for n in N_active)
                    print(
                        "t=", t,
                        "I_unfit:", val(self.I_unfit[(i, t, w0)]),
                        "Banknote Inventory Value:", voh,
                        "Total:", val(self.I_unfit[(i, t, w0)]) + voh,
                        "90% Cap:", 0.9 * self.Cap[i],
                    )

        # ---------------- POLICY PARAMETERS ----------------
        print("\n=== POLICY PARAMETERS (s, S) ===")
        for i in I:
            for b in B:
                for n in N_active:
                    s_val = val(self.s_var[(b, i, n)])
                    S_val = val(self.S_var[(b, i, n)])
                    s_lo = self.network.rdps[i].get_lower_bound(b, n)  # same concept as Pyomo (unscaled)
                    print(
                        f"RDP={i}, Denom={b}, NoteType={n}: "
                        f"s={float_fmt.format(s_val)}, "
                        f"S={float_fmt.format(S_val)}, "
                        f"s_lo={float_fmt.format(s_lo)}"
                    )

        print("\n=== Total Big S assignment compared to Capacity ===")
        for i in I:
            total_bigS = sum(val(self.S_var[(b, i, n)]) * b for b in B for n in N_active)
            print(f"RDP={i}, Total Big S value: {total_bigS}, 90% Capacity={0.9*self.Cap[i]}")

        # ---------------- TRIGGERS ----------------
        print("\n=== TRIGGERS (h, y, O) ===")
        for i in I:
            O_series = [val(self.O[(i, t, w0)]) for t in T]
            nz_O = nz_series(O_series)
            if nz_O != "all 0":
                print(f"RDP={i}, O: {nz_O}")

            for b in B:
                for n in N_active:
                    h_series = []
                    for t in T:
                        if (i, t) in self.IT_review:
                            h_val = val(self.h[(b, i, t, n, w0)])
                        else:
                            h_val = 0.0
                        h_series.append(h_val)

                    if any(abs(v) > tol for v in h_series):
                        print(f"RDP={i}, Denom={b}, NoteType={n}, h: {nz_series(h_series)}")

            # replenishment days based on joint_h (only on IT_review)
            repl_days = sorted({
                t for (i2, t) in self.IT_review
                if i2 == i and val(self.joint_h[(i2, t, w0)]) > 0.5
            })
            print(f"RDP={i} → {len(repl_days)} distinct replenishment days: {repl_days}")

            total_repl = sum(
                val(self.h[(b, i, t, n, w0)])
                for b in B for n in N_active for t in repl_days
            )
            print(f"RDP={i} → Total distinct replenishments: {total_repl}")

        # ---------------- ROUND TRIP ----------------
        if hasattr(self, "round_trip"):
            print("\n=== ROUND TRIP ===")
            for i in I:
                rt_series = [val(self.round_trip[(i, t, w0)]) for t in T]
                nz_rt = nz_series(rt_series)
                if nz_rt != "all 0":
                    print(f"RDP={i}, round_trip: {nz_rt}")

        # ---------------- SHIPMENT DETAILS (like Pyomo) ----------------
        print("\n=== SHIPMENT DETAILS ===")
        for (i, t) in self.IT_review:
            if val(self.joint_h[(i, t, w0)]) > 0.5:
                total_notes = sum(val(self.q[(b, i, t, n, w0)]) for b in B for n in N_active)
                total_value = sum(val(self.q[(b, i, t, n, w0)]) * b for b in B for n in N_active)
                max_value = self.max_shipment[i]
                max_notes = self.max_bags[i] * self.num_notes_in_bag
                print(
                    f"RDP={i}, t={t}: "
                    f"Total Notes Shipped={float_fmt.format(total_notes)}, "
                    f"Max Notes={float_fmt.format(max_notes)}, "
                    f"Total Value Shipped={float_fmt.format(total_value)}, "
                    f"Max Value={float_fmt.format(max_value)}"
                )

        print("\n=== Callback details ===\n")
        for (i, t) in self.IT_review:
            if val(self.O[(i, t, w0)]) > 0.5:
                u_val = val(self.U[(i, t, w0)])
                w_vals = {b: val(self.w[(b, i, t, w0)]) for b in B}
                max_value = self.max_shipment[i]
                max_notes = self.max_bags[i] * self.num_notes_in_bag
                total_callback_value = u_val + sum(w_vals[b] * b for b in B)
                total_callback_notes = sum(w_vals[b] for b in B) + int(u_val / 20.0)
                print(
                    f"RDP={i}, t={t}: "
                    f"Callback Unfit Value={float_fmt.format(u_val)}, "
                    f"Callback Notes={total_callback_notes}, "
                    f"Max Notes={float_fmt.format(max_notes)}, "
                    f"Total Callback Value={float_fmt.format(total_callback_value)}, "
                    f"Max Value={float_fmt.format(max_value)}"
                )
        # ---------------- STORE POLICY PARAMETERS BACK INTO RDP ----------------
        for i in I:
            rdp = self.network.rdps[i]

            if not hasattr(rdp, "opt_S"):
                rdp.opt_S = {}
            if not hasattr(rdp, "opt_s"):
                rdp.opt_s = {}

            for b in B:
                for n in N_active:
                    s_val = val(self.s_var[(b, i, n)])
                    S_val = val(self.S_var[(b, i, n)])

                    # scale back up (same as Pyomo)
                    rdp.opt_S[(b, n)] = S_val * self.scale_factor
                    rdp.opt_s[(b, n)] = s_val * self.scale_factor



    # def solve(self, mip_gap=0.0, verbose=True):
    #     if mip_gap is not None:
    #         #self.solver.SetSolverSpecificParametersAsString(f"ratioGap={mip_gap}")
    #         self.params.SetDoubleParam(
    #             pywraplp.MPSolverParameters.RELATIVE_MIP_GAP, float(mip_gap)
    #         )
    #     if not verbose:
    #         self.solver.EnableOutput(False)
    #     return self.solver.Solve()