"""
Headless debug simulation — runs p.DIRECT, auto-triggers 4WD→2WD at t=2s.
Logs full state every 50ms; dumps a condensed timeline and diagnosis at end.
"""

import math
import time
import sys
import numpy as np
import pybullet as p
import pybullet_data

# ---------------------------------------------------------------------------
# Import the same CONFIG / helpers used by tribot_sim.py
# ---------------------------------------------------------------------------
sys.path.insert(0, '.')
from tribot_sim import CONFIG, TribotBalanceBot, create_terrain
from plotjuggler_udp import PlotJugglerStreamer

# ---------------------------------------------------------------------------
# Override: headless, longer run, faster ramp for quicker diagnosis
# ---------------------------------------------------------------------------
CONFIG['CONTROLLER']  = 'lqr_ext'
CONFIG['SIM_DURATION'] = 12.0
CONFIG['ADD_SENSOR_NOISE'] = False

SWITCH_TIME = 3.0   # seconds before triggering 4WD→2WD

# ---------------------------------------------------------------------------

def run():
    physics_client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, CONFIG['GRAVITY'])
    p.setPhysicsEngineParameter(fixedTimeStep=CONFIG['TIMESTEP'], numSubSteps=1)

    create_terrain(CONFIG)
    print("\nLoading robot...")
    robot = TribotBalanceBot(physics_client, CONFIG)
    print("Robot loaded.\n")

    # -----------------------------------------------------------------------
    # Fine-grained per-step log buffer (ring, kept for last 5 s)
    # -----------------------------------------------------------------------
    LOG_DT   = 0.05   # keep one entry per 50 ms
    log_buf  = []
    switched = False
    switch_actual_time = None

    ctrl = robot.controller

    sim_time     = 0.0
    last_log     = -1.0

    print(f"{'t':>6}  {'pitch':>7}  {'trip_L':>7}  {'trip_R':>7}  "
          f"{'trip_ref':>9}  {'trip_err':>9}  "
          f"{'u_whl':>7}  {'u_trip':>7}  "
          f"{'ff_act':>6}  {'K_used':>8}  "
          f"{'trans':>5}")
    print("-" * 110)

    while sim_time < CONFIG['SIM_DURATION']:
        # Auto-trigger mode switch
        if not switched and sim_time >= SWITCH_TIME:
            robot.trigger_mode_switch()
            switch_actual_time = sim_time
            switched = True
            print(f"\n>>> MODE SWITCH triggered at t={sim_time:.3f}s\n")

        robot.update(sim_time, CONFIG['TIMESTEP'])
        p.stepSimulation()
        sim_time += CONFIG['TIMESTEP']

        # ---- sample at LOG_DT intervals ----
        if sim_time - last_log >= LOG_DT:
            last_log = sim_time

            dbg   = robot.get_debug_state()
            euler = dbg['euler_deg']    # (roll, pitch, yaw)
            ta    = dbg['triplet_ang']  # (L_rad, R_rad)
            tv    = dbg['triplet_vel']

            pitch_deg = euler[1]

            trip_ref  = getattr(ctrl, '_trip_ref',         0.0)
            trip_tgt  = getattr(ctrl, '_trip_ref_target',  0.0)
            trip_angle= getattr(ctrl, '_trip_angle',       0.0)
            trip_err  = getattr(ctrl, 'state_error_full',  np.zeros(6))[4]
            trans     = getattr(ctrl, 'transition_active', False)
            K_name    = ('TRANS' if (trans and ctrl.K_transition is not None
                                     and ctrl.K is ctrl.K_transition)
                         else ('AGG' if ctrl.aggressive_active else 'NORM'))

            u_whl   = ctrl.control_torque
            u_trip  = ctrl.triplet_torque_cmd
            ff_trip = 0.0  # can't read directly, but 'trans' active implies FF

            rec = dict(t=sim_time, pitch=pitch_deg,
                       ta_L=math.degrees(ta[0]), ta_R=math.degrees(ta[1]),
                       tv_L=tv[0], tv_R=tv[1],
                       trip_ref=math.degrees(trip_ref),
                       trip_err=math.degrees(trip_err),
                       u_whl=u_whl, u_trip=u_trip,
                       trans=int(trans), K=K_name,
                       act_L=robot.actual_torques[0], act_R=robot.actual_torques[1])
            log_buf.append(rec)

            state_err_full = getattr(ctrl, 'state_error_full', np.zeros(6))
            print(f"{sim_time:6.2f}  {pitch_deg:7.2f}  "
                  f"{math.degrees(ta[0]):7.2f}  {math.degrees(ta[1]):7.2f}  "
                  f"{math.degrees(trip_ref):9.3f}  {math.degrees(trip_err):9.3f}  "
                  f"{u_whl:7.3f}  {u_trip:7.3f}  "
                  f"{'Y' if trans else 'N':>6}  {K_name:>8}  "
                  f"{'TRANS' if trans else '     '}")

        if robot.check_fallen():
            print(f"\n!!! ROBOT FELL at t={sim_time:.3f}s  pitch={pitch_deg:.1f}° !!!\n")
            break

    # -----------------------------------------------------------------------
    # Post-mortem analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("POST-MORTEM ANALYSIS")
    print("=" * 80)

    if not log_buf:
        print("No data recorded.")
        p.disconnect()
        return

    pre  = [r for r in log_buf if r['t'] < SWITCH_TIME]
    post = [r for r in log_buf if r['t'] >= SWITCH_TIME]

    def stats(arr, key):
        vals = [r[key] for r in arr]
        if not vals:
            return 0, 0, 0
        return np.mean(vals), np.std(vals), max(abs(v) for v in vals)

    print(f"\n--- PRE-SWITCH (0 → {SWITCH_TIME:.1f}s) ---")
    m, s, mx = stats(pre, 'pitch')
    print(f"  Pitch:     mean={m:.2f}° std={s:.2f}° max|={mx:.2f}°")
    m, s, mx = stats(pre, 'u_whl')
    print(f"  Whl cmd:   mean={m:.3f} std={s:.3f} max|={mx:.3f} Nm")
    if pre:
        print(f"  Trip ref:  {pre[-1]['trip_ref']:.2f}°  (should be 0° in 4WD)")
        print(f"  Trip L/R:  {pre[-1]['ta_L']:.2f}° / {pre[-1]['ta_R']:.2f}°")

    print(f"\n--- POST-SWITCH ({SWITCH_TIME:.1f}s → end) ---")
    if post:
        m, s, mx = stats(post, 'pitch')
        print(f"  Pitch:     mean={m:.2f}° std={s:.2f}° max|={mx:.2f}°")
        m, s, mx = stats(post, 'u_whl')
        print(f"  Whl cmd:   mean={m:.3f} std={s:.3f} max|={mx:.3f} Nm")
        m, s, mx = stats(post, 'u_trip')
        print(f"  Trip cmd:  mean={m:.3f} std={s:.3f} max|={mx:.3f} Nm")

        # Find when triplet first reaches target
        TARGET_DEG = math.degrees(math.pi / 3.0)
        reached = [r for r in post if abs(r['ta_L'] - TARGET_DEG) < 5.0]
        if reached:
            print(f"  Trip L reached ±5° of target ({TARGET_DEG:.1f}°) at t={reached[0]['t']:.2f}s")
        else:
            last = post[-1]
            print(f"  Trip L NEVER reached target {TARGET_DEG:.1f}°  "
                  f"(ended at {last['ta_L']:.1f}°)")

        # Check if K_transition was ever used
        k_trans_used = [r for r in post if r['K'] == 'TRANS']
        print(f"  K_TRANS samples: {len(k_trans_used)}/{len(post)}")

        # Max pitch during transition
        trans_only = [r for r in post if r['trans']]
        if trans_only:
            max_pitch = max(abs(r['pitch']) for r in trans_only)
            print(f"  Max |pitch| during transition: {max_pitch:.2f}°")

    # -----------------------------------------------------------------------
    # Key diagnosis checks
    # -----------------------------------------------------------------------
    print("\n--- DIAGNOSIS ---")

    # 1. Is the robot even stable before the switch?
    if pre:
        max_pre = max(abs(r['pitch']) for r in pre)
        print(f"  [{'OK' if max_pre < 5 else 'FAIL'}] Pre-switch max pitch = {max_pre:.2f}°  "
              f"(expect <5°)")

    # 2. Does the triplet reference actually advance?
    if post:
        refs = [r['trip_ref'] for r in post]
        print(f"  [{'OK' if max(refs) > 5 else 'FAIL'}] Trip ref max = {max(refs):.2f}°  "
              f"(expect up to {math.degrees(math.pi/3):.1f}°)")

    # 3. Does the triplet angle actually move?
    if post:
        angles_L = [r['ta_L'] for r in post]
        travel = max(angles_L) - min(angles_L)
        print(f"  [{'OK' if travel > 5 else 'FAIL'}] Trip L joint travel = {travel:.2f}°  "
              f"(expect ~60°)")

    # 4. Does pitch exceed safety limit?
    if post:
        safety_deg = math.degrees(CONFIG.get('ELQR_PITCH_SAFETY_LIMIT', 0.35))
        max_pitch_post = max(abs(r['pitch']) for r in post)
        print(f"  [{'WARN' if max_pitch_post > safety_deg else 'OK'}] "
              f"Max post-switch pitch = {max_pitch_post:.2f}°  "
              f"(safety={safety_deg:.1f}°)")

    # 5. Is torque actually being applied to triplet?
    if post:
        trip_cmds = [abs(r['u_trip']) for r in post]
        max_trip = max(trip_cmds) if trip_cmds else 0
        mean_trip = np.mean(trip_cmds) if trip_cmds else 0
        print(f"  [{'OK' if max_trip > 0.1 else 'FAIL'}] "
              f"Max |triplet cmd| = {max_trip:.3f} Nm  mean={mean_trip:.3f} Nm")

    # 6. Sign check: trip_ref and ta_L should both go positive
    if post:
        trip_ref_end = post[-1]['trip_ref']
        ta_L_end     = post[-1]['ta_L']
        same_sign    = (trip_ref_end > 0) == (ta_L_end > 0)
        print(f"  [{'OK' if same_sign else 'FAIL'}] "
              f"Sign check: trip_ref={trip_ref_end:.1f}°  ta_L={ta_L_end:.1f}°  "
              f"(should both be positive for 2WD)")

    # 7. Is B matrix sign consistent? (quick analytical check)
    from control_lqr_extended import build_extended_state_space
    A, B = build_extended_state_space(CONFIG)
    print(f"\n  B[3,1] (pitch from τ_trip) = {B[3,1]:.4f}   "
          f"(negative = triplet push pitches body forward — must be handled by FF)")
    print(f"  B[5,1] (trip_rate from τ_trip) = {B[5,1]:.4f}  "
          f"(positive = triplet torque drives triplet — correct)")
    print(f"  Wheel FF comp ratio = {-B[3,1]/B[3,0]:.4f}")

    print("\n--- FULL LOG (last 20 samples) ---")
    print(f"{'t':>6}  {'pitch':>7}  {'ta_L':>7}  {'ta_R':>7}  "
          f"{'ref':>7}  {'err':>7}  {'u_whl':>7}  {'u_trip':>7}  "
          f"{'K':>5}  {'T':>1}")
    for r in log_buf[-20:]:
        print(f"{r['t']:6.2f}  {r['pitch']:7.2f}  {r['ta_L']:7.2f}  {r['ta_R']:7.2f}  "
              f"{r['trip_ref']:7.2f}  {r['trip_err']:7.2f}  {r['u_whl']:7.3f}  {r['u_trip']:7.3f}  "
              f"{r['K']:>5}  {r['trans']:>1}")

    p.disconnect()
    print("\nDone.")


if __name__ == '__main__':
    run()
