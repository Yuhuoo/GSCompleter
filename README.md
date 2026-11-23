# GSCompleter

## Project structure

```text
GSCompleter/
├── README.md
├── .gitmodules
├── PE-Field/          # GSCompleter-PE-Field, branch: gscompleter
└── src/
    ├── DepthSplat/    # GSCompleter-DepthSplat, branch: gs_completer_v1
    └── ADGaussian/    # GSCompleter-ADGaussian, branch: gscompleter
```

The three projects are managed as Git submodules. Their tracking branches are recorded in `.gitmodules`; the parent repository pins each submodule to a specific commit.

## Clone

```bash
git clone --recurse-submodules https://github.com/Yuhuoo/GSCompleter.git
cd GSCompleter
```

For an existing clone:

```bash
git submodule update --init --recursive
```

To explicitly update the three projects to the latest commits on their configured branches:

```bash
git submodule update --remote src/DepthSplat src/ADGaussian PE-Field
```

After updating a submodule, commit its new pointer in the parent repository to share the updated version.
