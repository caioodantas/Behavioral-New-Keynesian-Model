"""
```
eqcond(m::XGabaix)
```

Expresses the equilibrium conditions in canonical form using Γ0, Γ1, C, Ψ, and Π matrices.
Using the mappings of states/equations to integers defined in gabaif.jl, coefficients are
specified in their proper positions.

### Outputs

* `Γ0` (`n_states` x `n_states`) holds coefficients of current time states.
* `Γ1` (`n_states` x `n_states`) holds coefficients of lagged states.
* `C`  (`n_states` x `1`) is a vector of constants
* `Ψ`  (`n_states` x `n_shocks_exogenous`) holds coefficients of iid shocks.
* `Π`  (`n_states` x `n_states_expectational`) holds coefficients of expectational states.
"""
function eqcond(m::XGabaix)
    endo = m.endogenous_states
    exo  = m.exogenous_shocks
    ex   = m.expected_shocks
    eq   = m.equilibrium_conditions

    Γ0 = zeros(n_states(m), n_states(m))
    Γ1 = zeros(n_states(m), n_states(m))
    C  = zeros(n_states(m))
    Ψ  = zeros(n_states(m), n_shocks_exogenous(m))
    Π  = zeros(n_states(m), n_shocks_expectational(m))

    ### ENDOGENOUS STATES ###

    ### 1. Consumption Euler Equation

    Γ0[eq[:eq_euler], endo[:x_t]]  = 1.
    Γ0[eq[:eq_euler], endo[:i_t]] = 1*m[:β]/m[:γ]
    Γ0[eq[:eq_euler], endo[:η_d_t]] = -1
    Γ0[eq[:eq_euler], endo[:Ex_t1]] = -1*m[:M]
    Γ0[eq[:eq_euler], endo[:Eπ_t1]] = -1*m[:β]/m[:γ]

    ### 2. NK Phillips Curve

    Γ0[eq[:eq_phillips], endo[:x_t]] = -(-1+1/m[:β])*(1-m[:β]*m[:θ])*(m[:γ]+m[:ϕ])
    Γ0[eq[:eq_phillips], endo[:π_t]] = 1
    Γ0[eq[:eq_phillips], endo[:ϵ_s_t]] = -1
    Γ0[eq[:eq_phillips], endo[:Eπ_t1]] = -m[:β]*m[:M]*(m[:θ]+((1-m[:β]*m[:θ])/(1-m[:β]*m[:θ]*m[:M]))*(1-m[:θ]))

    ### 3. Monetary Policy Rule

    Γ0[eq[:eq_mp], endo[:x_t]] = -(1-m[:ρ_i])*m[:ϕ_x]
    Γ0[eq[:eq_mp], endo[:π_t]] = -(1-m[:ρ_i])*m[:ϕ_π]
    Γ0[eq[:eq_mp], endo[:i_t]] = 1
    Γ0[eq[:eq_mp], endo[:η_m_t]] = -1
    Γ1[eq[:eq_mp], endo[:i_t]] = m[:ρ_i]

    ### 4. Output lag

    Γ0[eq[:eq_x_t1], endo[:x_t1]] = 1
    Γ1[eq[:eq_x_t1], endo[:x_t]] = 1

    ### 5. Demand disturbance

    Γ0[eq[:eq_η_d], endo[:η_d_t]] = 1
    Γ1[eq[:eq_η_d], endo[:η_d_t]] = m[:ρ_d]
    Ψ[eq[:eq_η_d], exo[:ϵ_d_t]] = 1

    ### 6. Monetary Policy disturbance

    Γ0[eq[:eq_η_m], endo[:η_m_t]] = 1
    Γ1[eq[:eq_η_m], endo[:η_m_t]] = m[:ρ_m]
    Ψ[eq[:eq_η_m], exo[:ϵ_m_t]] = 1

    ### 7. Expected output

    Γ0[eq[:eq_Ex], endo[:x_t]] = 1
    Γ1[eq[:eq_Ex], endo[:Ex_t1]] = 1
    Π[eq[:eq_Ex], ex[:Ex_sh]] = 1

    ### 8. Expected inflation

    Γ0[eq[:eq_Eπ], endo[:π_t]] = 1
    Γ1[eq[:eq_Eπ], endo[:Eπ_t1]] = 1
    Π[eq[:eq_Eπ], ex[:Eπ_sh]] = 1

    return Γ0, Γ1, C, Ψ, Π
end
