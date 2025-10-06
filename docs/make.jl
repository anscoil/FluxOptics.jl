using FluxOptics
using Documenter
using Makie

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
                 #         "api/optical_components/core/core.md"
                 #     ],
                 #     "Free-Space Propagators" => [
                 #         "api/optical_components/freespace/index.md",
                 #         "api/optical_components/freespace/freespace.md"
                 #     ],
                 #     "Bulk Propagators" => [
                 #         "api/optical_components/bulk/index.md",
                 #         "api/optical_components/bulk/bulk.md"
                 #     ],
                 #     "Active Media" => [
                 #         "api/optical_components/active/index.md",
                 #         "api/optical_components/active/active_media.md"
                 #     ]
                 # ],
                 "OptimisersExt" => [
                     "api/optimisers/index.md",
                     "API" => "api/optimisers/optimisers.md"
                 ],
                 "Metrics" => [
                     "api/metrics/index.md",
                     "API" => "api/metrics/metrics.md"
                 ],
                 "Plotting" => [
                     "api/plotting/index.md",
                     "API" => "api/plotting/plotting.md"
                 ]
             ]
         ])

deploydocs(; repo = "github.com/anscoil/FluxOptics.jl", devbranch = "main")
