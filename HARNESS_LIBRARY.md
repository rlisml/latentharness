# HARNESS_LIBRARY.md - Behavioral harness text registry

Machine-read registry consumed by `harness/harness_text.py`. Single source of
truth for harness text: training and evaluation code must load texts from here,
never hardcode them. Hashes are over the exact UTF-8 bytes of the text with no
trailing newline (`printf '%s' '<text>' | sha256sum`); any wording change is a
NEW entry with a new id and hash, never an in-place edit.

---

## Entry: VERIFY-001

| Field | Value |
|---|---|
| **harness text** | `Before answering, solve the problem, then explicitly verify your answer by independently re-checking the key step(s). If verification fails, correct your answer. End with "Verification: <one-sentence check>" followed by "Answer: <final answer>."` |
| **sha256** | `3f89f806bfdc03bda340a880df6bb87d9e5d3d0b88ed3a1cc8b6b3ff42f19413` |
| **status** | `harmful-control` |

## Entry: DECOMPOSE-001

| Field | Value |
|---|---|
| **harness text** | `Decompose the problem into numbered sub-steps before solving. Solve each sub-step explicitly, then combine them for the final result. Format each sub-step as "Step k: <sub-problem> -> <result>". End with "Answer: <final answer>."` |
| **sha256** | `37933fef5c67314356d0ebde58645a57c8d8726f7633e0af6b498aa2a5187ddd` |
| **status** | `distilled` |

## Entry: COMPUTE-001

| Field | Value |
|---|---|
| **harness text** | `Do not guess or pick from answer choices. First write the arithmetic expression that answers the question from the given numbers, then evaluate it step by step, one operation per line. End with "Answer: <final number>."` |
| **sha256** | `6ca36929fed6dacd0f0707a71c4c7e6a1d5b31c57e6e3565e3cd3fb44601fe71` |
| **status** | `screened-out` |

## Entry: CONCISE-001

| Field | Value |
|---|---|
| **harness text** | `Answer compactly: state the plan in one short sentence, do the computation, and give the answer. Do not restate the problem and do not repeat yourself. End with "Answer: <final answer>."` |
| **sha256** | `595bfcb9957c8133464e131cb8434216a820d5baec6d6418e7eb9cebb217917d` |
| **status** | `harmful-control` |

## Entry: ELIMINATE-001

| Field | Value |
|---|---|
| **harness text** | `Consider each answer option one at a time: state whether it fits the question and drop the ones that do not. Choose the single option that remains and give its letter with its text. End with "Answer: (<letter>) <option text>."` |
| **sha256** | `afd323f5feeff9f0168c2c27540c2d16259a943530c66a45eead29770ad19bf7` |
| **status** | `distilled` |

## Entry: QUOTE-001

| Field | Value |
|---|---|
| **harness text** | `Before answering, find the one sentence in the passage that decides the question and begin your reply by quoting it exactly. Then state whether the question is true or false given that sentence. End with "Answer: true." or "Answer: false."` |
| **sha256** | `01082df92e04c9dba2aebea39d4d435d5f5c7d92893248099c94991d6c0a22b6` |
| **status** | `distilled` |

## Entry: TESTCONSULT-001

| Field | Value |
|---|---|
| **harness text** | `Before writing code, check the function name and signature the shown test cases call, and use exactly that interface. Solve only this task, and end your reply with [DONE] immediately after your solution — do not write any further tasks or code.` |
| **sha256** | `37ee945121ead8af701e8ab8ac7bea5bbfc51f8c2fa27db29b39558c50ca181d` |
| **status** | `screened-in` |

## Entry: DECOMPOSE-002

| Field | Value |
|---|---|
| **harness text** | `Before tackling the problem, explicitly decompose it into numbered sub-steps and create a step-by-step plan. Then, solve each sub-step separately and format the solution as 'Step k: <sub-problem> -> <result>'. Finally, combine the solutions to find the final answer. Ensure the final answer is clearly stated with 'Answer: <final answer>'.` |
| **sha256** | `a77d25943c1c3a8e5f45c30842a7a564f944264bfd7d73cf1c365bac2b31b7c3` |
| **status** | `screened-in` |

## Entry: DECOMPOSE-003

| Field | Value |
|---|---|
| **harness text** | `To ensure a step-by-step solution, decompose the problem into numbered sub-steps before proceeding. Solve each sub-step explicitly, write down the result in the format 'Step k: <sub-problem> -> <result>', and only then combine the results to reach the final answer. Avoid any shortcuts.` |
| **sha256** | `433f48de263629c0a02211aeeb8500465b8ada1f28bbfed21ca4669ad91c73f6` |
| **status** | `screened-in` |

## Entry: COMPUTE-002

| Field | Value |
|---|---|
| **harness text** | `Do not guess. Write down the exact arithmetic expression from the given numbers, then evaluate it step by step, one operation per line, and end with the final number. This will be your answer.` |
| **sha256** | `64111b54455dca7ac5142b9b312497e6a4f2392a5cddb655b697ea61379a77ae` |
| **status** | `screened-out` |

## Entry: COMPUTE-003

| Field | Value |
|---|---|
| **harness text** | `Before answering, create a step-by-step arithmetic expression from the given numbers, evaluating one operation per line until you reach the final number. This is your answer.` |
| **sha256** | `056aa2bcded813f820fe9092280220f443536487dccb8a0e258a904fc93add97` |
| **status** | `screened-out` |

## Entry: CONCISE-002

| Field | Value |
|---|---|
| **harness text** | `State the plan briefly, do the computation, and give the answer in one final sentence ending with 'Answer: <final answer>'. Do not repeat yourself.` |
| **sha256** | `ee41696694bcdd8c6bd9dc1a9d6adcc03a0f4962c35a73c86e65f63fbefcb195` |
| **status** | `distilled` |

## Entry: CONCISE-003

| Field | Value |
|---|---|
| **harness text** | `State your plan, perform the necessary computation, and clearly conclude with the answer in the format 'Answer: <final answer>'. Do not reiterate the problem or previous steps.` |
| **sha256** | `66129b3d63eddfd218e47bc0c4a25a4159b4956aafbf6cf71690c68fd9a2dc32` |
| **status** | `screened-out` |

## Entry: CONCISE-004

| Field | Value |
|---|---|
| **harness text** | `State the plan briefly, do the computation, and give the answer in one final sentence with the correct code format ending with 'Answer: <final answer>'. Do not repeat yourself.` |
| **sha256** | `cf4e37a8d88be28dc80b0cc7b51cc7bf3fc9a52eed27def02712eb584a8d7c73` |
| **status** | `distilled` |

## Entry: CONCISE-005

| Field | Value |
|---|---|
| **harness text** | `Before computing, state your plan in a few words, do the computation, and give the answer as a single sentence ending with 'Answer: <final answer>'.` |
| **sha256** | `e3df47720f0ceed8cf24acbe4a0b7dbaa06fdc053b86b7b282761ae4597bc601` |
| **status** | `screened-out` |

## Entry: SINGLETASK-001

| Field | Value |
|---|---|
| **harness text** | `Solve only the one given task. Work out the answer once, then output exactly one final line "Answer: <final answer>" and stop — do not write any further tasks, questions, options, or repeated working after it. Never invent answer options that were not given.` |
| **sha256** | `bbc7caab84c3794f9e78f2bc8fb8109a7ff1a5187f063ba34a6fe3fb7583af20` |
| **status** | `screened-in` |

## Entry: SINGLETASK-002

| Field | Value |
|---|---|
| **harness text** | `Commit to exactly one of the given options and end with "Answer: <letter>". Do not invent extra options or letters, do not restate the option list, and stop immediately after the Answer line.` |
| **sha256** | `1f81231890d1581df2b309ad72133b2c7653e3872bbc773879891ad06ac2790e` |
| **status** | `screened-out` |
