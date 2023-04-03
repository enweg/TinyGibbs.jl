using TinyGibbs
using Documenter

DocMeta.setdocmeta!(TinyGibbs, :DocTestSetup, :(using TinyGibbs); recursive=true)

makedocs(;
    modules=[TinyGibbs],
    authors="Enrico Wegner",
    repo="https://github.com/enweg/TinyGibbs.jl/blob/{commit}{path}#{line}",
    sitename="TinyGibbs.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://enweg.github.io/TinyGibbs.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/enweg/TinyGibbs.jl",
    devbranch="main",
)
