# %%
import xtrack as xt

RING = xt.Line.from_json('lattice_v1o5.json')

# %%
import json, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import xobjects as xo
import xpart as xp

# ERROR function

def apply_alignment_and_field_errors(
    RING,
    seed=42,                                                                                                        #later assigning random seed values
    quad_dx_rms=0.3e-3, quad_dy_rms=0.3e-3, quad_roll_rms=0.3e-3, quad_k1_err_rms=1e-3,
    rbend_dx_rms=0.3e-3, rbend_dy_rms=0.3e-3, rbend_roll_rms=0.3e-3, rbend_angle_err_rms=0.3e-3,
    skip_wigglers_in_rbends=True,
    wiggler_name_token="wig.",
    output_dir="errors"
):
    print(f"[info] CWD: {os.getcwd()}")
    rng = np.random.default_rng(seed)
    error_log = []
    n_quads, n_rbends, n_skipped_wigs = 0, 0, 0

    for name, elem in zip(RING.element_names, RING.elements):

        # Quadrupoles
        if isinstance(elem, xt.Quadrupole):
            dx   = rng.normal(scale=quad_dx_rms)
            dy   = rng.normal(scale=quad_dy_rms)
            roll = rng.normal(scale=quad_roll_rms)
            dk1  = rng.normal(scale=quad_k1_err_rms)

            if not hasattr(elem, "_k1_nominal"):
                elem._k1_nominal = float(elem.k1)

            elem.shift_x = dx
            elem.shift_y = dy
            elem.rotate_z = roll
            elem.k1 = elem._k1_nominal * (1.0 + dk1)

            error_log.append({'name': name,'type': 'Quadrupole','dx': dx,'dy': dy,'roll': roll,'dk1': dk1})
            n_quads += 1

        # RBends (skip wigglers that contain 'wig.' exactly, case-insensitive)
        elif isinstance(elem, xt.RBend):
            is_wiggler_like = skip_wigglers_in_rbends and (wiggler_name_token.lower() in name.lower())
            if is_wiggler_like:
                error_log.append({'name': name, 'type': 'RBend(SKIPPED_WIGGLER)'})
                n_skipped_wigs += 1
                continue

            dx     = rng.normal(scale=rbend_dx_rms)
            dy     = rng.normal(scale=rbend_dy_rms)
            roll   = rng.normal(scale=rbend_roll_rms)
            dangle = rng.normal(scale=rbend_angle_err_rms)

            if not hasattr(elem, "_angle_nominal"):
                elem._angle_nominal = float(elem.angle)

            elem.shift_x = dx
            elem.shift_y = dy
            elem.rotate_z = roll
            elem.angle = elem._angle_nominal * (1.0 + dangle)

            error_log.append({'name': name,'type': 'RBend','dx': dx,'dy': dy,'roll': roll,'dangle': dangle})
            n_rbends += 1

    # --- save (always) ---
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    error_filename = Path(output_dir) / f"errors_seed_{seed}.json"
    with open(error_filename, "w") as f:
        json.dump({
            'seed': seed,
            'summary': {'quads_modified': n_quads, 'rbends_modified': n_rbends, 'rbends_skipped_wigglers': n_skipped_wigs},
            'entries': error_log
        }, f, indent=2)

    print(f"[done] Quads: {n_quads}, RBends: {n_rbends}, Skipped wigglers: {n_skipped_wigs}")
    print(f"[saved] {error_filename.resolve()}")
    print()
    return error_log, str(error_filename.resolve())


# %%
seed_val = np.random.randint(1, 100001)  # upper bound is exclusive

SEED_DIR = os.path.join(OUTPUT_DIR, f"seed_{seed_val}")

if not os.path.exists(SEED_DIR):
    os.makedirs(SEED_DIR)

seed_res = {}
error_log, path = apply_alignment_and_field_errors(RING, seed=seed_val, wiggler_name_token="wig")

# Spin tracking for seed #

tw_spin = RING.twiss(polarization=True)

seed_res[seed_val] = {
      'spin_polarization_eq': tw_spin.spin_polarization_eq*100,
      'spin_polarization_inf_no_depol': tw_spin.spin_polarization_inf_no_depol*100,
      'buildup_time': tw_spin.spin_t_pol_buildup_s,
      'spin_x': tw_spin.spin_x,
      'spin_y': tw_spin.spin_y,
      'spin_z': tw_spin.spin_z
  }

print(seed_res[seed_val])
# %%


RING.configure_radiation(model='mean')
tw = RING.twiss(eneloss_and_damping=True)



# Generate a matched bunch distribution
np.random.seed(0)
particles = xp.generate_matched_gaussian_bunch(
    line=RING,
    nemitt_x=tw.eq_nemitt_x,
    nemitt_y=0.01*(tw.eq_nemitt_x),  # Assume 1% coupling for vertical emittance
    sigma_z=np.sqrt(tw.eq_gemitt_zeta * tw.bets0),
    num_particles=300,
    engine='linear')

# Add stable phase
particles.zeta += tw.zeta[0]
particles.delta += tw.delta[0]

# Initialize spin of all particles along n0
particles.spin_x = tw_spin.spin_x[0]
particles.spin_y = tw_spin.spin_y[0]
particles.spin_z = tw_spin.spin_z[0]

RING.configure_spin('auto' )
RING.configure_radiation(model='quantum')

# Enable parallelization
RING.discard_tracker()
RING.build_tracker(_context=xo.ContextCpu(omp_num_threads=10))

# Track
num_turns=20000
RING.track(particles, num_turns=num_turns, turn_by_turn_monitor=True,
           with_progress=10)
mon = RING.record_last_track


# fit depolarization time


mask_alive = mon.state > 0
pol_x = mon.spin_x.sum(axis=0)/mask_alive.sum(axis=0)
pol_y = mon.spin_y.sum(axis=0)/mask_alive.sum(axis=0)
pol_z = mon.spin_z.sum(axis=0)/mask_alive.sum(axis=0)
pol = np.sqrt(pol_x**2 + pol_y**2 + pol_z**2)

i_start = 100 # Skip a few turns (small initial mismatch)
pol_to_fit = pol[i_start:]/pol[i_start]
 
# Fit depolarization time (linear fit)
from scipy.stats import linregress
turns = np.arange(len(pol_to_fit))
slope, intercept, r_value, p_value, std_err = linregress(turns, pol_to_fit)
# Calculate depolarization time
t_dep_turns = -1 / slope


plt.figure()
plt.plot(pol_to_fit-1, label='Tracking')
plt.plot(turns, intercept*np.exp(-turns/t_dep_turns) - 1, label='Fit')
plt.ylabel(r'$P/P_0 - 1$')
plt.xlabel('Turn')
plt.subplots_adjust(left=.2)
plt.savefig(os.path.join(SEED_DIR, "polarization_fit.png"), dpi=1000)
plt.legend()


print(f"Depolarization time in turns: {t_dep_turns}")
print(f"Depolarization time in seconds: {t_dep_turns * tw.T_rev0} s")


p_inf = tw_spin['spin_polarization_inf_no_depol']
t_pol_turns = tw_spin['spin_t_pol_component_s']/tw.T_rev0


p_eq = p_inf * 1 / (1 + t_pol_turns/t_dep_turns)
print(f'Equilibrium polarization: {p_eq*100:.2f} %')


# -------------------------
# Save seed_res variables
# -------------------------
with open(os.path.join(SEED_DIR, "seed_res.dat"), "w") as f:
    f.write(f"spin_polarization_eq {tw_spin.spin_polarization_eq * 100}\n")
    f.write(f"spin_polarization_inf_no_depol {tw_spin.spin_polarization_inf_no_depol * 100}\n")
    f.write(f"buildup_time {tw_spin.spin_t_pol_buildup_s}\n")
    f.write("spin_x " + " ".join(map(str, tw_spin.spin_x)) + "\n")
    f.write("spin_y " + " ".join(map(str, tw_spin.spin_y)) + "\n")
    f.write("spin_z " + " ".join(map(str, tw_spin.spin_z)) + "\n")


with open(os.path.join(SEED_DIR, "polarization_results.dat"), "w") as f:
    f.write(f"pol_x {pol_x}\n")
    f.write(f"pol_y {pol_y}\n")
    f.write(f"pol_z {pol_z}\n")
    f.write(f"pol {pol}\n")
    f.write(f"t_dep_turns {t_dep_turns}\n")
    f.write(f"t_dep_seconds {t_dep_turns * tw.T_rev0}\n")
    f.write(f"p_inf {p_inf}\n")
    f.write(f"t_pol_turns {t_pol_turns}\n")
    f.write(f"p_eq_percent {p_eq * 100}\n")


# %%



