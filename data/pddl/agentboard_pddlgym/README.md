# AgentBoard PDDL Backend

`pddlgym/` is an unmodified copy of the PDDLGym package distributed with
[AgentBoard](https://github.com/hkust-nlp/AgentBoard), including the PDDL
domains and problems the benchmark uses.

Source:

```text
hkust-nlp/AgentBoard, commit bb7255e2daf1989069a186dad9e53f70680961db
agentboard/environment/pddl_env/pddlgym
```

The PDDL benchmark uses these registered environments of the package:

```text
PDDLEnvGripper-v0      20 problems
PDDLEnvBlockworld-v0   10 problems
PDDLEnvBarman-v0       20 problems
PDDLEnvTyreworld-v0    10 problems
```

`data/pddl/env/pddl_env.py` loads this copy and rejects the PyPI `pddlgym`
package, so do not install or mix the two.

## Licenses

This directory keeps its upstream licenses; the repository's `LICENSE` does not
apply to it.

- AgentBoard states that its code is licensed under Apache-2.0 and its data under
  GPL-2.0 (README of the AgentBoard repository, which has no `LICENSE` file).
- PDDLGym, from which AgentBoard's copy is derived, is licensed under the MIT
  license (Copyright (c) 2020 Tom Silver and Rohan Chitnis).

Full texts: `LICENSES/Apache-2.0.txt`, `LICENSES/GPL-2.0.txt`, and
`LICENSES/PDDLGym-MIT.txt`.
