using DifferentialEquations
using Plots

# --- Two-Layer Climate Model ODE ---
function two_layer_climate_model!(dT, T, p, t)
    S0, C_a, C_o, A, B0, Fmax, k = p
    T_a, T_o = T

    # --- Feedbacks ---
    # Ice–albedo feedback
    albedo = 0.3 - 0.01 * (T_a - 288)
    albedo = clamp(albedo, 0.1, 0.7)

    # Water vapor feedback
    B = B0 - 0.01 * (T_a - 288)
    B = max(B, 0.5)

    # Cloud feedback (simplified: modifies OLR intercept)
    A_eff = A + 0.5 * (T_a - 288)

    # --- Radiation Balance ---
    # Seasonal cycle: ±2% variation in solar input
    ASR = (S0 * (1 - albedo) / 4) * (1 + 0.02 * sin(2π * t / 1.0))

    # --- Forcing Terms ---
    # Ramp forcing (anthropogenic CO₂ increase over 200 years)
    F_ramp = t <= 200 ? Fmax * (t / 200) : Fmax

    # Volcanic cooling events
    F_volcano = (t >= 100 && t <= 105) ? -2.0 : 0.0
    F_volcano += (t >= 600 && t <= 605) ? -3.0 : 0.0

    # Solar variability (11-year cycle, ±0.5 W/m²)
    F_solar = 0.5 * sin(2π * t / 11.0)

    # Natural variability (random noise)
    F_noise = 0.3 * randn()

    # Total forcing
    F = F_ramp + F_volcano + F_solar + F_noise

    # Outgoing Longwave Radiation
    OLR = A_eff + B * T_a

    # --- ODEs ---
    dT_a = (ASR + F - OLR) / C_a - k * (T_a - T_o) / C_a
    dT_o = k * (T_a - T_o) / C_o

    dT[1] = dT_a
    dT[2] = dT_o
end

println("Two-layer climate model with all feedbacks and forcings defined.")

# --- Parameters ---
S0   = 1361.0       # Solar constant (W/m^2)
C_a  = 1.0e8        # Atmosphere + mixed layer heat capacity
C_o  = 1.0e10       # Deep ocean heat capacity
A    = -337.825     # OLR intercept
B0   = 2.0          # Base OLR slope
Fmax = 3.7          # Max radiative forcing (W/m^2) ~ CO₂ doubling
k    = 1.0e7        # Coupling strength between layers

p = (S0, C_a, C_o, A, B0, Fmax, k)

# --- Initial Conditions ---
T_a0 = 288.0
T_o0 = 288.0
T0   = [T_a0, T_o0]

tspan = (0.0, 1000.0)

# --- Solve ---
prob = ODEProblem(two_layer_climate_model!, T0, tspan, p)
sol = solve(prob, Tsit5())

println("ODE problem solved.")

# --- Plot Atmosphere and Ocean Temperatures ---
p1 = plot(sol.t, [u[1] for u in sol.u],
          xlabel="Time (Years)", ylabel="Temperature (K)",
          title="Atmosphere vs Deep Ocean Temperatures",
          label="Atmosphere/Mixed Layer")

plot!(sol.t, [u[2] for u in sol.u],
      label="Deep Ocean")

display(p1)

# --- Plot Temperature Difference (Atmosphere - Ocean) ---
temp_diff = [u[1] - u[2] for u in sol.u]

p2 = plot(sol.t, temp_diff,
          xlabel="Time (Years)", ylabel="ΔT (K)",
          title="Atmosphere–Ocean Temperature Difference",
          legend=false)

display(p2)

# --- Climate Sensitivity Diagnostic ---
# Equilibrium Climate Sensitivity (ECS) in K per CO₂ doubling
ECS = Fmax / B0
println("Equilibrium Climate Sensitivity (ECS): ", ECS, " K per CO₂ doubling")
