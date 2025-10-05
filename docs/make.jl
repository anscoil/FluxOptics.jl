using FluxOptics
using Documenter

DocMeta.setdocmeta!(FluxOptics, :DocTestSetup, :(using FluxOptics);
                    recursive = true)

makedocs(;
         modules = [FluxOptics],
         authors = "Nicolas Barré",
         sitename = "FluxOptics.jl",
         format = Documenter.HTML(;
                                  prettyurls = get(ENV, "CI", "false") == "true",
                                  canonical = "https://anscoil.github.io/FluxOptics.jl",
                                  edit_link = "main",
                                  assets = String[]),
         doctest = false,
         checkdocs = :none, # :exports
         warnonly = true, #[:cross_references],
         pages = [
             "Home" => "index.md",
             "Reference" => [
                 "api/index.md",
                 "GridUtils" => [
                     "api/gridutils/index.md",
                     "API" => "api/gridutils/gridutils.md"
                 ],
                 "Modes" => [
                     "api/modes/index.md",
                     "API" => "api/modes/modes.md"
                 ],
                 "Fields" => [
                     "api/fields/index.md",
                     "API" => "api/fields/fields.md"
                 ],
                 # "Optical Components" => [
                 #     "api/optical_components/index.md",
                 #     "Core" => [
                 #         "api/optical_components/core/index.md",
                 #         "api/optical_components/core/types.md",
                 #         "api/optical_components/core/sources.md",
                 #         "api/optical_components/core/static.md",
                 #         "api/optical_components/core/systems.md"
                 #     ],
                 #     "Free-Space Propagators" => [
                 #         "api/optical_components/freespace/index.md",
                 #         "api/optical_components/freespace/angular_spectrum.md",
                 #         "api/optical_components/freespace/rayleigh_sommerfeld.md",
                 #         "api/optical_components/freespace/fourier.md",
                 #         "api/optical_components/freespace/rotation.md"
                 #     ],
                 #     "Bulk Propagators" => [
                 #         "api/optical_components/bulk/index.md",
                 #         "api/optical_components/bulk/bpm.md"
                 #     ],
                 #     "Active Media" => [
                 #         "api/optical_components/active/index.md",
                 #         "api/optical_components/active/gain.md"
                 #     ]
                 # ],
                 # "Optimizers" => [
                 #     "api/optimizers/index.md",
                 #     "api/optimizers/rules.md",
                 #     "api/optimizers/proximal_operators.md"
                 # ],
                 "Metrics" => [
                     "api/metrics/index.md",
                     "API" => "api/metrics/metrics.md"
                 ]
             ]])

deploydocs(; repo = "github.com/anscoil/FluxOptics.jl", devbranch = "main")
