using FluxOptics
using Documenter

DocMeta.setdocmeta!(FluxOptics, :DocTestSetup, :(using FluxOptics); recursive=true)

makedocs(;
    modules=[FluxOptics],
    authors="Nicolas Barré",
    sitename="FluxOptics.jl",
    format=Documenter.HTML(;
        canonical="https://anscoil.github.io/FluxOptics.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/anscoil/FluxOptics.jl",
    devbranch="main",
)
