"""
```
XGabaix{T} <: AbstractRepModel{T}
```

The `XGabaix` type defines the structure of the simple Behavioral New Keynesian DSGE
model described in 'A Behavioral New Keynesian Model' by Xavier Gabaix.

### Fields

#### Parameters and Steady-States
* `parameters::Vector{AbstractParameter}`: Vector of all time-invariant model
  parameters.

* `steady_state::Vector{AbstractParameter}`: Model steady-state values, computed
  as a function of elements of `parameters`.

* `keys::OrderedDict{Symbol,Int}`: Maps human-readable names for all model
  parameters and steady-states to their indices in `parameters` and
  `steady_state`.

#### Inputs to Measurement and Equilibrium Condition Equations

The following fields are dictionaries that map human-readable names to row and
column indices in the matrix representations of of the measurement equation and
equilibrium conditions.

* `endogenous_states::OrderedDict{Symbol,Int}`: Maps each state to a column in
  the measurement and equilibrium condition matrices.

* `exogenous_shocks::OrderedDict{Symbol,Int}`: Maps each shock to a column in
  the measurement and equilibrium condition matrices.

* `expected_shocks::OrderedDict{Symbol,Int}`: Maps each expected shock to a
  column in the measurement and equilibrium condition matrices.

* `equilibrium_conditions::OrderedDict{Symbol,Int}`: Maps each equlibrium
  condition to a row in the model's equilibrium condition matrices.

* `endogenous_states_augmented::OrderedDict{Symbol,Int}`: Maps lagged states to
  their columns in the measurement and equilibrium condition equations. These
  are added after `gensys` solves the model.

* `observables::OrderedDict{Symbol,Int}`: Maps each observable to a row in the
  model's measurement equation matrices.

* `pseudo_observables::OrderedDict{Symbol,Int}`: Maps each pseudo-observable to
  a row in the model's pseudo-measurement equation matrices.

#### Model Specifications and Settings

* `spec::String`: The model specification identifier, \"gabaix\", cached
  here for filepath computation.

* `subspec::String`: The model subspecification number, indicating that some
  parameters from the original model spec (\"ss0\") are initialized
  differently. Cached here for filepath computation.

* `settings::Dict{Symbol,Setting}`: Settings/flags that affect computation
  without changing the economic or mathematical setup of the model.

* `test_settings::Dict{Symbol,Setting}`: Settings/flags for testing mode

#### Other Fields

* `rng::MersenneTwister`: Random number generator. Can be is seeded to ensure
  reproducibility in algorithms that involve randomness (such as
  Metropolis-Hastings).

* `testing::Bool`: Indicates whether the model is in testing mode. If `true`,
  settings from `m.test_settings` are used in place of those in `m.settings`.

* `observable_mappings::OrderedDict{Symbol,Observable}`: A dictionary that
  stores data sources, series mnemonics, and transformations to/from model units.
  DSGE.jl will fetch data from the Federal Reserve Bank of
  St. Louis's FRED database; all other data must be downloaded by the
  user. See `load_data` and `Observable` for further details.

* `pseudo_observable_mappings::OrderedDict{Symbol,PseudoObservable}`: A
  dictionary that stores names and transformations to/from model units. See
  `PseudoObservable` for further details.
"""
mutable struct XGabaix{T} <: AbstractRepModel{T}
    parameters::ParameterVector{T}                         # vector of all time-invariant model parameters
    steady_state::ParameterVector{T}                       # model steady-state values
    keys::OrderedDict{Symbol,Int}                          # human-readable names for all the model
                                                           # parameters and steady-states

    endogenous_states::OrderedDict{Symbol,Int}             # these fields used to create matrices in the
    exogenous_shocks::OrderedDict{Symbol,Int}              # measurement and equilibrium condition equations.
    expected_shocks::OrderedDict{Symbol,Int}               #
    equilibrium_conditions::OrderedDict{Symbol,Int}        #
    endogenous_states_augmented::OrderedDict{Symbol,Int}   #
    observables::OrderedDict{Symbol,Int}                   #
    pseudo_observables::OrderedDict{Symbol,Int}            #

    spec::String                                           # Model specification number (eg "m990")
    subspec::String                                        # Model subspecification (eg "ss0")
    settings::Dict{Symbol,Setting}                         # Settings/flags for computation
    test_settings::Dict{Symbol,Setting}                    # Settings/flags for testing mode
    rng::MersenneTwister                                   # Random number generator
    testing::Bool                                          # Whether we are in testing mode or not

    observable_mappings::OrderedDict{Symbol, Observable}
    pseudo_observable_mappings::OrderedDict{Symbol, PseudoObservable}
end

description(m::XGabaix) = "Julia implementation of model defined in 'A Behavioral New Keynesian Model' by Xavier Gabaix: XGabaix, $(m.subspec)"

"""
`init_model_indices!(m::gabaix)`

Arguments:
`m:: gabaix`: a model object

Description:
Initializes indices for all of `m`'s states, shocks, and equilibrium conditions.
"""
function init_model_indices!(m::XGabaix)
    # Endogenous states
    endogenous_states = collect([
        :x_t, :π_t, :i_t, :x_t1, :η_d_t, :η_m_t, :Ex_t1, :Eπ_t1])

    # Exogenous shocks
    exogenous_shocks = collect([
        :ϵ_d_t, :ϵ_m_t, :ϵ_s_t])

    # Expectations shocks
    expected_shocks = collect([
        :Ex_sh, :Eπ_sh])

    # Equilibrium conditions
    equilibrium_conditions = collect([
        :eq_euler, :eq_phillips, :eq_mp, :eq_x_t1, :eq_η_d, :eq_η_m, :eq_Ex, :eq_Eπ])

    # Additional states added after solving model
    # Lagged states and observables measurement error
    endogenous_states_augmented = []

    # Observables
    observables = keys(m.observable_mappings)

    # Pseudo-observables
    pseudo_observables = keys(m.pseudo_observable_mappings)

    for (i,k) in enumerate(endogenous_states);           m.endogenous_states[k]           = i end
    for (i,k) in enumerate(exogenous_shocks);            m.exogenous_shocks[k]            = i end
    for (i,k) in enumerate(expected_shocks);             m.expected_shocks[k]             = i end
    for (i,k) in enumerate(equilibrium_conditions);      m.equilibrium_conditions[k]      = i end
    for (i,k) in enumerate(endogenous_states);           m.endogenous_states[k]           = i end
    for (i,k) in enumerate(endogenous_states_augmented); m.endogenous_states_augmented[k] = i+length(endogenous_states) end
    for (i,k) in enumerate(observables);                 m.observables[k]                 = i end
    for (i,k) in enumerate(pseudo_observables);          m.pseudo_observables[k]          = i end
end


function XGabaix(subspec::String="ss0";
                       custom_settings::Dict{Symbol, Setting} = Dict{Symbol, Setting}(),
                       testing = false)

    # Model-specific specifications
    spec               = split(basename(@__FILE__),'.')[1]
    subspec            = subspec
    settings           = Dict{Symbol,Setting}()
    test_settings      = Dict{Symbol,Setting}()
    rng                = MersenneTwister(0)

    # initialize empty model
    m = XGabaix{Float64}(
            # model parameters and steady state values
            Vector{AbstractParameter{Float64}}(), Vector{Float64}(), OrderedDict{Symbol,Int}(),

            # model indices
            OrderedDict{Symbol,Int}(), OrderedDict{Symbol,Int}(), OrderedDict{Symbol,Int}(), OrderedDict{Symbol,Int}(), OrderedDict{Symbol,Int}(), OrderedDict{Symbol,Int}(), OrderedDict{Symbol,Int}(),

            spec,
            subspec,
            settings,
            test_settings,
            rng,
            testing,
            OrderedDict{Symbol,Observable}(),
            OrderedDict{Symbol,PseudoObservable}())

    # Set settings
    model_settings!(m)
    default_test_settings!(m)
    for custom_setting in values(custom_settings)
        m <= custom_setting
    end

    # Set observable and pseudo-observable transformations
    init_observable_mappings!(m)
    init_pseudo_observable_mappings!(m)

    # Initialize parameters
    init_parameters!(m)

    init_model_indices!(m)
    init_subspec!(m)
    steadystate!(m)

    return m
end

"""
```
init_parameters!(m::XGabaix)
```

Initializes the model's parameters, as well as empty values for the steady-state
parameters (in preparation for `steadystate!(m)` being called to initialize
those).
"""
function init_parameters!(m::XGabaix)
    # Initialize parameters
    m <= parameter(:γ, 1, (1e-3, 10), (1e-3, 10), ModelConstructors.Exponential(), RootInverseGamma(1, 0.75), fixed=false,
                   description="γ: Risk aversion.",
                   tex_label="\\gamma")

    m <= parameter(:M, 0.9, (0.01, 0.9999), (0.01, 0.9999), ModelConstructors.Exponential(), BetaAlt(0.8, 0.15), fixed=false,
                    description="M: Base inattention.",
                    tex_label="M")

    m <= parameter(:ϕ, 1, fixed=true,
                   description="ϕ: Survival rate of prices.",
                   tex_label="\\phi")

    m <= parameter(:θ, 0.875, fixed=true,
                    description="θ: Inverse Frisch elasticity.",
                    tex_label="\\theta")

    m <= parameter(:ϕ_π, 1.5, (0, 3), (0, 3), ModelConstructors.Exponential(), Normal(1.5, 0.5), fixed=false,
                   description="ϕ_1: The weight on inflation in the monetary policy rule.",
                   tex_label="\\phi_\\pi")
    m <= parameter(:ϕ_x, 0.25, (0, 3), (0, 3), ModelConstructors.Exponential(), Normal(1.5, 0.5), fixed=false,
                   description="ϕ_2: The weight on the output gap in the monetary policy rule.",
                   tex_label="\\phi_x")

    m <= parameter(:β, 0.99, fixed=true,
                   description="β: Discount factor.",
                   tex_label="\\beta")

    m <= parameter(:ρ_i, 0.8, (0.01, 0.9999), (0.01, 0.9999), ModelConstructors.Untransformed(), BetaAlt(0.7,0.1), fixed=false,
                   description="ρ_i: AR(1) coefficient on interest rate.",
                   tex_label="\\rho_i")

    m <= parameter(:ρ_d, 0.5, (0.01, 0.9999), (0.01, 0.9999), ModelConstructors.Untransformed(), BetaAlt(0.5,0.2), fixed=false,
                   description="ρ_d: AR(1) coefficient on shocks to GDP.",
                   tex_label="\\rho_d")

    m <= parameter(:ρ_m, 0.5, (0.01, 0.9999), (0.01, 0.9999), ModelConstructors.Untransformed(), BetaAlt(0.5,0.2), fixed=false,
                   description="ρ_m: AR(1) coefficient on shocks to inflation.",
                   tex_label="\\rho_m")

    m <= parameter(:σ_m, 0.3, (0.001, 10), (0.001, 10), ModelConstructors.Exponential(), RootInverseGamma(0.3, 1), fixed=false,
                   description="σ_m: Standard deviation of shocks to monetary rule.",
                   tex_label="\\sigma_m")

    m <= parameter(:σ_d, 0.3, (0.001, 10), (0.001, 10), ModelConstructors.Exponential(), RootInverseGamma(0.3, 1), fixed=false,
                   description="σ_d: Standard deviation of shocks to GDP.",
                   tex_label="\\sigma_d")

    m <= parameter(:σ_s, 0.3, (0.001, 10), (0.001, 10), ModelConstructors.Exponential(), RootInverseGamma(0.3, 1), fixed=false,
                   description="σ_s: Standard deviation of shocks to inflation.",
                   tex_label="\\sigma_s")

    m <= parameter(:e_y, 0.20*0.579923, fixed=true,
                                  description="e_x: Measurement error on GDP growth.",
                                  tex_label="e_x")

    m <= parameter(:e_π, 0.20*1.470832, fixed=true,
                                  description="e_π: Measurement error on inflation.",
                                  tex_label="e_\\pi")

    m <= parameter(:e_R, 0.20*2.237937, fixed=true,
                                  description="e_i: Measurement error on the interest rate.",
                                  tex_label="e_i")

    m <= parameter(:γ_Q, 1, fixed=true,

                                  description="γ_Q: Steady state growth rate of technology.",
                                  tex_label="\\gamma_Q")

    m <= parameter(:π_star,  0, fixed=true,
                                  description="π_star: Target inflation rate.",
                                  tex_label="\\pi*")

    m <= parameter(:rA, 4.04, fixed=true,
                                  description="rA: β (discount factor) = 1/(1+ rA/400).",
                                  tex_label="rA")
end

"""
```
steadystate!(m::XGabaix)
```

Calculates the model's steady-state values. `steadystate!(m)` must be called whenever
the parameters of `m` are updated.
"""
function steadystate!(m::XGabaix)
    return m
end

function model_settings!(m::XGabaix)
    default_settings!(m)

    # Data
    m <= Setting(:data_id, 0, "Dataset identifier")
    m <= Setting(:cond_full_names, [:obs_gdp, :obs_nominalrate],
        "Observables used in conditional forecasts")
    m <= Setting(:cond_semi_names, [:obs_nominalrate],
        "Observables used in semiconditional forecasts")

    # Metropolis-Hastings
    m <= Setting(:mh_cc, 0.27,
                 "Jump size for Metropolis-Hastings (after initialization)")

    # Forecast
    m <= Setting(:use_population_forecast, true,
                 "Whether to use population forecasts as data")
    m <= Setting(:forecast_zlb_value, 0.13,
        "Value of the zero lower bound in forecast periods, if we choose to enforce it")
end

function shock_groupings(m::XGabaix)
    gov = ShockGroup("η_d", [:ϵ_m_t], RGB(0.70, 0.13, 0.13)) # firebrick
    tfp = ShockGroup("η_m", [:ϵ_m_t], RGB(1.0, 0.55, 0.0)) # darkorange
    pol = ShockGroup("pol", vcat([:ϵ_s_t], [Symbol("ϵ_s_tl$i") for i = 1:n_anticipated_shocks(m)]),
                     RGB(1.0, 0.84, 0.0)) # gold
    det = ShockGroup("dt", [:dettrend], :gray40)

    return [gov, tfp, pol, det]
end
