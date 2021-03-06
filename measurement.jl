"""
```
measurement(m::XGabaix{T}, TTT::Matrix{T},
            RRR::Matrix{T}, CCC::Vector{T}) where {T<:AbstractFloat}
```

Assign measurement equation

```
x_t = ZZ*s_t + DD + u_t
```

where

```
Var(ϵ_t) = QQ
Var(u_t) = EE
Cov(ϵ_t, u_t) = 0
```
"""
function measurement(m::XGabaix{T},
                     TTT::Matrix{T},
                     RRR::Matrix{T},
                     CCC::Vector{T}) where {T<:AbstractFloat}
    endo     = m.endogenous_states
    endo_new = m.endogenous_states_augmented
    exo      = m.exogenous_shocks
    obs      = m.observables

    _n_observables = n_observables(m)
    _n_states = n_states_augmented(m)
    _n_shocks_exogenous = n_shocks_exogenous(m)

    ZZ = zeros(_n_observables, _n_states)
    DD = zeros(_n_observables)
    EE = zeros(_n_observables, _n_observables)
    QQ = zeros(_n_shocks_exogenous, _n_shocks_exogenous)

    ## Output growth
    ZZ[obs[:obs_gdp], endo[:x_t]]  = 1.0
    ZZ[obs[:obs_gdp], endo[:x_t1]] = -1.0
    ZZ[obs[:obs_gdp], endo[:η_m_t]]  = 1.0
    DD[obs[:obs_gdp]]              = m[:γ_Q]

    ## Inflation
    ZZ[obs[:obs_cpi], endo[:π_t]] = 4.0
    DD[obs[:obs_cpi]]             = m[:π_star]

    ## Federal Funds Rate
    ZZ[obs[:obs_nominalrate], endo[:i_t]] = 4.0
    DD[obs[:obs_nominalrate]]             = m[:π_star] + m[:rA] + 4.0*m[:γ_Q]

    # Measurement error
    EE[obs[:obs_gdp], endo[:x_t]]         = m[:e_x]^2
    EE[obs[:obs_cpi], endo[:π_t]]         = m[:e_π]^2
    EE[obs[:obs_nominalrate], endo[:i_t]] = m[:e_i]^2

    # Variance of innovations
    QQ[exo[:ϵ_m_t],exo[:ϵ_m_t]]   = (m[:σ_m])^2
    QQ[exo[:ϵ_d_t],exo[:ϵ_d_t]]   = (m[:σ_d])^2
    QQ[exo[:rm_sh],exo[:rm_sh]] = (m[:σ_s])^2

    return Measurement(ZZ, DD, QQ, EE)
end
