using Documenter: DocMeta, doctest
using FluxOptics
using Statistics
using Test

filters = [r"[0-9]+-element.*"]

DocMeta.setdocmeta!(FluxOptics, :DocTestSetup, :(using FluxOptics); recursive = true)

doctest(FluxOptics; doctestfilters = filters)
