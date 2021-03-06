function init_pseudo_observable_mappings!(m::XGabaix)

    pseudo_names = [:x_t, :π_t, :η_m_t, :NominalFFR, :RealFFR]

    # Create PseudoObservable objects
    pseudo = OrderedDict{Symbol,PseudoObservable}()
    for k in pseudo_names
        pseudo[k] = PseudoObservable(k)
    end

    # Fill in names and reverse transforms
    pseudo[:x_t].name = "Output Growth"
    pseudo[:x_t].longname = "Output Growth Per Capita"

    pseudo[:π_t].name = "Inflation"
    pseudo[:π_t].longname = "Inflation"
    pseudo[:π_t].rev_transform = quartertoannual

    pseudo[:η_m_t].name     = "η_m_t"
    pseudo[:η_m_t].longname = "η_m_t"

    pseudo[:NominalFFR].name     = "Nominal FFR"
    pseudo[:NominalFFR].longname = "Nominal FFR at an annual rate"

    pseudo[:RealFFR].name     = "Real FFR"
    pseudo[:RealFFR].longname = "Real FFR at an annual rate"

    # Add to model object
    m.pseudo_observable_mappings = pseudo
end