# Third-Party Notices

The repository's `LICENSE` covers its own code. The components below keep their
own licenses.

## ConMem (adapted code)

Parts of this repository, including the multi-agent host implementations,
benchmark environments, and evaluation scripts, are adapted from the ConMem
code release (https://anonymous.4open.science/r/ConMemCode), which is
distributed under the following license.

```
MIT License

Copyright (c) 2026 FA66666

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## AgentBoard PDDLGym and PDDL problems (bundled, unmodified)

`data/pddl/agentboard_pddlgym/pddlgym/` is an unmodified copy of
`agentboard/environment/pddl_env/pddlgym` from AgentBoard
(https://github.com/hkust-nlp/AgentBoard, commit
bb7255e2daf1989069a186dad9e53f70680961db). AgentBoard states that its code is
licensed under the Apache License 2.0 and its dataset under the GNU General
Public License, version 2. The package derives from PDDLGym
(https://github.com/tomsilver/pddlgym, MIT license, Copyright (c) 2020 Tom
Silver and Rohan Chitnis).

Full license texts are in `data/pddl/agentboard_pddlgym/LICENSES/`
(`Apache-2.0.txt`, `GPL-2.0.txt`, `PDDLGym-MIT.txt`).

## AgentBoard PDDL test items (bundled)

`data/pddl/test.jsonl` contains the 60 PDDL test items of the AgentBoard
dataset (https://huggingface.co/datasets/hkust-nlp/agentboard, GPL-2.0), with
subgoals stored as a list and the subtask name added to each item. License text:
`data/pddl/agentboard_pddlgym/LICENSES/GPL-2.0.txt`.

## Downloaded at run time (not included)

- KodCode-Light-RL-10K: CC BY-NC 4.0.
- TriviaQA: Apache 2.0. The [upstream README](https://github.com/mandarjoshi90/triviaqa#readme)
  explicitly states that the license applies to both code and data; its
  Hugging Face card currently lists `unknown`. The upstream statement is the
  basis for the license description here.
- PopQA: distributed through an MIT-licensed repository
  (https://github.com/AlexTMallen/adaptive-retrieval). The Hugging Face card
  does not specify a separate data license; the repository license should not
  be interpreted as blanket permission for all underlying third-party content.
- Search-R1 retrieval server (Apache 2.0) and its e5 index over the 2018
  Wikipedia dump, which derives from Wikipedia text available under CC BY-SA:
  see https://github.com/PeterGriffinJin/Search-R1.
