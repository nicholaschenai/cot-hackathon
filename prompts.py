PROMPT = """<|start_header_id|>user<|end_header_id|>\n\nSolve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\nProblem: Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n## Step 1: Consider the expression inside the first square root\nThe expression inside the first square root is $x-2$, and it must be non-negative. This means that $x-2 \\ge 0$, which simplifies to $x \\ge 2$.\n\n## Step 2: Consider the expression inside the second square root\nThe expression inside the second square root is $5-x$, and it must be non-negative. This means that $5-x \\ge 0$, which simplifies to $x \\le 5$.\n\n## Step 3: Consider the denominator of the expression\nThe denominator of the expression is $\\sqrt{5-x}$, and it cannot be equal to zero. This means that $5-x>0$, which simplifies to $x<5$.\n\n## Step 4: Combine the results from the previous steps\nCombining the results from the previous steps, we have $x \\ge 2$ and $x < 5$. This means that the domain of the expression is $[2,5)$.\n\nThe final answer is: $\\boxed{[2,5)}$. I hope it is correct.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSolve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\nProblem: If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n## Step 1: Understand the properties of determinants\nWe know that the determinant of a product of two matrices is equal to the product of their determinants.\n\n## Step 2: Apply the property to the given matrices\nUsing the property, we can write $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B})$.\n\n## Step 3: Substitute the given values\nWe are given that $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12$. Substituting these values, we get $\\det (\\mathbf{A} \\mathbf{B}) = (2)(12)$.\n\n## Step 4: Calculate the final answer\nEvaluating the product, we find that $\\det (\\mathbf{A} \\mathbf{B}) = 24$.\n\nThe final answer is: $\\boxed{24}$. I hope it is correct.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSolve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\nProblem: Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n## Step 1: Calculate the total weight Terrell lifts with two 20-pound weights\nTerrell lifts two 20-pound weights 12 times, so the total weight he lifts is $2 \\cdot 12 \\cdot 20 = 480$ pounds.\n\n## Step 2: Calculate the total weight Terrell lifts with two 15-pound weights for n times\nIf Terrell lifts two 15-pound weights instead for $n$ times, he will lift a total of $2 \\cdot 15 \\cdot n = 30n$ pounds of weight.\n\n## Step 3: Equate the total weight lifted with 15-pound weights to 480 pounds and solve for n\nEquating the total weight lifted with 15-pound weights to 480 pounds, we can solve for $n$: $30n = 480$.\n\n## Step 4: Solve for n\nTo solve for $n$, divide both sides by 30: $n = \\frac{480}{30}$.\n\n## Step 5: Calculate the value of n\nThe value of $n$ is $\\frac{480}{30} = 16$.\n\nThe final answer is: $\\boxed{16}$. I hope it is correct.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSolve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\nProblem: If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n## Step 1: Understand the problem\nWe are given a system of linear equations with two variables, $x$ and $y$, and two constants, $a$ and $b$. Our goal is to find the ratio $\\frac{a}{b}$, assuming $b$ is nonzero.\n\n## Step 2: Multiply the first equation by a suitable factor\nWe multiply the first equation by $-\\frac{3}{2}$ to obtain an equation with the same coefficients for $y$ as the second equation: $$-\\frac{3}{2}(6x-4y)=a\\Rightarrow 6y-9x=-\\frac{3}{2}a.$$\n\n## Step 3: Equate the right-hand sides of the two equations\nSince we know that $6y-9x=b$ from the second equation, we can equate the right-hand sides of the two equations to obtain $$-\\frac{3}{2}a=b.$$\n\n## Step 4: Solve for the ratio $\\frac{a}{b}$\nFinally, we can solve for the ratio $\\frac{a}{b}$ by dividing both sides of the equation by $b$, giving us $$\\frac{a}{b}=-\\frac{2}{3}.$$\n\nThe final answer is: $\\boxed{-\\frac{2}{3}}. I hope it is correct.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSolve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\nProblem: {problem}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""

ALT_PROMPT = """<|start_header_id|>user<|end_header_id|>\n\nSolve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\nProblem: Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n## Step 1: Consider the expression inside the first square root\nThe expression inside the first square root is $x-2$, and it must be non-negative. This means that $x-2 \\ge 0$, which simplifies to $x \\ge 2$.\n\n## Step 2: Consider the expression inside the second square root\nThe expression inside the second square root is $5-x$, and it must be non-negative. This means that $5-x \\ge 0$, which simplifies to $x \\le 5$.\n\n## Step 3: Consider the denominator of the expression\nThe denominator of the expression is $\\sqrt{5-x}$, and it cannot be equal to zero. This means that $5-x>0$, which simplifies to $x<5$.\n\n## Step 4: Combine the results from the previous steps\nCombining the results from the previous steps, we have $x \\ge 2$ and $x < 5$. This means that the domain of the expression is $[2,5)$.\n\nThe final answer is: $\\boxed{[2,5)}$. I hope it is correct.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSolve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\nProblem: If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n## Step 1: Understand the properties of determinants\nWe know that the determinant of a product of two matrices is equal to the product of their determinants.\n\n## Step 2: Apply the property to the given matrices\nUsing the property, we can write $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B})$.\n\n## Step 3: Substitute the given values\nWe are given that $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12$. Substituting these values, we get $\\det (\\mathbf{A} \\mathbf{B}) = (2)(12)$.\n\n## Step 4: Calculate the final answer\nEvaluating the product, we find that $\\det (\\mathbf{A} \\mathbf{B}) = 24$.\n\nThe final answer is: $\\boxed{24}$. I hope it is correct.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSolve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\nProblem: Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n## Step 1: Calculate the total weight Terrell lifts with two 20-pound weights\nTerrell lifts two 20-pound weights 12 times, so the total weight he lifts is $2 \\cdot 12 \\cdot 20 = 480$ pounds.\n\n## Step 2: Calculate the total weight Terrell lifts with two 15-pound weights for n times\nIf Terrell lifts two 15-pound weights instead for $n$ times, he will lift a total of $2 \\cdot 15 \\cdot n = 30n$ pounds of weight.\n\n## Step 3: Equate the total weight lifted with 15-pound weights to 480 pounds and solve for n\nEquating the total weight lifted with 15-pound weights to 480 pounds, we can solve for $n$: $30n = 480$.\n\n## Step 4: Solve for n\nTo solve for $n$, divide both sides by 30: $n = \\frac{480}{30}$.\n\n## Step 5: Calculate the value of n\nThe value of $n$ is $\\frac{480}{30} = 16$.\n\nThe final answer is: $\\boxed{16}$. I hope it is correct.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSolve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\nProblem: If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n## Step 1: Understand the problem\nWe are given a system of linear equations with two variables, $x$ and $y$, and two constants, $a$ and $b$. Our goal is to find the ratio $\\frac{a}{b}$, assuming $b$ is nonzero.\n\n## Step 2: Multiply the first equation by a suitable factor\nWe multiply the first equation by $-\\frac{3}{2}$ to obtain an equation with the same coefficients for $y$ as the second equation: $$-\\frac{3}{2}(6x-4y)=a\\Rightarrow 6y-9x=-\\frac{3}{2}a.$$\n\n## Step 3: Equate the right-hand sides of the two equations\nSince we know that $6y-9x=b$ from the second equation, we can equate the right-hand sides of the two equations to obtain $$-\\frac{3}{2}a=b.$$\n\n## Step 4: Solve for the ratio $\\frac{a}{b}$\nFinally, we can solve for the ratio $\\frac{a}{b}$ by dividing both sides of the equation by $b$, giving us $$\\frac{a}{b}=-\\frac{2}{3}.$$\n\nThe final answer is: $\\boxed{-\\frac{2}{3}}. I hope it is correct.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSolve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.\n\nProblem: {problem}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""

TIDIED_PROMPT = """
<|start_header_id|>user<|end_header_id|>

Solve the following math problem efficiently and clearly:

- For simple problems (2 steps or fewer):
  Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
  Use this step-by-step format:

  ## Step 1: [Concise description]
  [Brief explanation and calculations]

  ## Step 2: [Concise description]
  [Brief explanation and calculations]

  ...

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.

Where [answer] is just the final number or expression that solves the problem.

Problem: Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

## Step 1: Consider the expression inside the first square root
The expression inside the first square root is $x-2$, and it must be non-negative. This means that $x-2 \\ge 0$, which simplifies to $x \\ge 2$.

## Step 2: Consider the expression inside the second square root
The expression inside the second square root is $5-x$, and it must be non-negative. This means that $5-x \\ge 0$, which simplifies to $x \\le 5$.

## Step 3: Consider the denominator of the expression
The denominator of the expression is $\\sqrt{5-x}$, and it cannot be equal to zero. This means that $5-x>0$, which simplifies to $x<5$.

## Step 4: Combine the results from the previous steps
Combining the results from the previous steps, we have $x \\ge 2$ and $x < 5$. This means that the domain of the expression is $[2,5)$.

The final answer is: $\\boxed{[2,5)}$. I hope it is correct.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Solve the following math problem efficiently and clearly:

- For simple problems (2 steps or fewer):
  Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
  Use this step-by-step format:

  ## Step 1: [Concise description]
  [Brief explanation and calculations]

  ## Step 2: [Concise description]
  [Brief explanation and calculations]

  ...

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.

Where [answer] is just the final number or expression that solves the problem.

Problem: If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B})$.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

## Step 1: Understand the properties of determinants
We know that the determinant of a product of two matrices is equal to the product of their determinants.

## Step 2: Apply the property to the given matrices
Using the property, we can write $\\det (\\mathbf{A} \\mathbf{B}) = (\\det \\mathbf{A})(\\det \\mathbf{B})$.

## Step 3: Substitute the given values
We are given that $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12$. Substituting these values, we get $\\det (\\mathbf{A} \\mathbf{B}) = (2)(12)$.

## Step 4: Calculate the final answer
Evaluating the product, we find that $\\det (\\mathbf{A} \\mathbf{B}) = 24$.

The final answer is: $\\boxed{24}$. I hope it is correct.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Solve the following math problem efficiently and clearly:

- For simple problems (2 steps or fewer):
  Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
  Use this step-by-step format:

  ## Step 1: [Concise description]
  [Brief explanation and calculations]

  ## Step 2: [Concise description]
  [Brief explanation and calculations]

  ...

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.

Where [answer] is just the final number or expression that solves the problem.

Problem: Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

## Step 1: Calculate the total weight Terrell lifts with two 20-pound weights
Terrell lifts two 20-pound weights 12 times, so the total weight he lifts is $2 \\cdot 12 \\cdot 20 = 480$ pounds.

## Step 2: Calculate the total weight Terrell lifts with two 15-pound weights for n times
If Terrell lifts two 15-pound weights instead for $n$ times, he will lift a total of $2 \\cdot 15 \\cdot n = 30n$ pounds of weight.

## Step 3: Equate the total weight lifted with 15-pound weights to 480 pounds and solve for n
Equating the total weight lifted with 15-pound weights to 480 pounds, we can solve for $n$: $30n = 480$.

## Step 4: Solve for n
To solve for $n$, divide both sides by 30: $n = \\frac{480}{30}$.

## Step 5: Calculate the value of n
The value of $n$ is $\\frac{480}{30} = 16$.

The final answer is: $\\boxed{16}$. I hope it is correct.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Solve the following math problem efficiently and clearly:

- For simple problems (2 steps or fewer):
  Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
  Use this step-by-step format:

  ## Step 1: [Concise description]
  [Brief explanation and calculations]

  ## Step 2: [Concise description]
  [Brief explanation and calculations]

  ...

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.

Where [answer] is just the final number or expression that solves the problem.

Problem: If the system of equations

\\begin{align*}
6x-4y&=a,\\\\
6y-9x &=b.
\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

## Step 1: Understand the problem
We are given a system of linear equations with two variables, $x$ and $y$, and two constants, $a$ and $b$. Our goal is to find the ratio $\\frac{a}{b}$, assuming $b$ is nonzero.

## Step 2: Multiply the first equation by a suitable factor
We multiply the first equation by $-\\frac{3}{2}$ to obtain an equation with the same coefficients for $y$ as the second equation: $$-\\frac{3}{2}(6x-4y)=a\\Rightarrow 6y-9x=-\\frac{3}{2}a.$$

## Step 3: Equate the right-hand sides of the two equations
Since we know that $6y-9x=b$ from the second equation, we can equate the right-hand sides of the two equations to obtain $$-\\frac{3}{2}a=b.$$

## Step 4: Solve for the ratio $\\frac{a}{b}$
Finally, we can solve for the ratio $\\frac{a}{b}$ by dividing both sides of the equation by $b$, giving us $$\\frac{a}{b}=-\\frac{2}{3}.$$

The final answer is: $\\boxed{-\\frac{2}{3}}. I hope it is correct.
<|eot_id|><|start_header_id|>user<|end_header_id|>

Solve the following math problem efficiently and clearly:

- For simple problems (2 steps or fewer):
  Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
  Use this step-by-step format:

  ## Step 1: [Concise description]
  [Brief explanation and calculations]

  ## Step 2: [Concise description]
  [Brief explanation and calculations]

  ...

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.

Where [answer] is just the final number or expression that solves the problem.

Problem: {problem}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""


LLAMA_SYS_PROMPT = """

Solve the following math problem efficiently and clearly:

- For simple problems (2 steps or fewer):
  Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
  Use this step-by-step format:

  ## Step 1: [Concise description]
  [Brief explanation and calculations]

  ## Step 2: [Concise description]
  [Brief explanation and calculations]

  ...

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.

Where [answer] is just the final number or expression that solves the problem.

You may use symbolic math tools if necessary.
You do not need to solve everything in one go: You can use the `sympy_execute` tool to execute code step by step.



Example Problem: Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.

## Step 1: Consider the expression inside the first square root
The expression inside the first square root is $x-2$, and it must be non-negative. This means that $x-2 \\ge 0$, which simplifies to $x \\ge 2$.

## Step 2: Consider the expression inside the second square root
The expression inside the second square root is $5-x$, and it must be non-negative. This means that $5-x \\ge 0$, which simplifies to $x \\le 5$.

## Step 3: Consider the denominator of the expression
The denominator of the expression is $\\sqrt{5-x}$, and it cannot be equal to zero. This means that $5-x>0$, which simplifies to $x<5$.

## Step 4: Combine the results from the previous steps
Combining the results from the previous steps, we have $x \\ge 2$ and $x < 5$. This means that the domain of the expression is $[2,5)$.

The final answer is: $\\boxed{[2,5)}$. I hope it is correct.

"""

OUTPUT_PROMPT = """
When you are done, conclude with:

Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.

Where [answer] is just the final number or expression that solves the problem.
"""



OAI_REACT_INSTRUCTION = """
## Instruction
You will be given a math problem.

Do not solve the problem in one go. Rather, solve it step by step.
Your next reply should either be a new step, or a conclusion that solves the entire problem.

For each step, first write down your thoughts from the previous observation, 
then use the `sympy_execute` tool to execute sympy code.
"""

OAI_SYS_PROMPT = OAI_REACT_INSTRUCTION + OUTPUT_PROMPT


OAI_ZERO_SHOT_PROMPT = """
Solve the following math problem efficiently and clearly:

- For simple problems (2 steps or fewer):
  Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
  Use this step-by-step format:

  ## Step 1: [Concise description]
  [Brief explanation and calculations]

  ## Step 2: [Concise description]
  [Brief explanation and calculations]

  ...

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.

Where [answer] is just the final number or expression that solves the problem.
"""

def get_system_prompt(option, strategy):
    if option == "remote":
        if strategy == "react":
            return OAI_SYS_PROMPT
        elif strategy == "zero_shot":
            return OAI_ZERO_SHOT_PROMPT
    else:
        return LLAMA_SYS_PROMPT
    
# backups

OAI_REACT_INSTRUCTION_BACKUP = """
## Instruction

Solve the following math problem efficiently and clearly:

You do not need to solve everything in one go:
You must use the `sympy_execute` tool to execute sympy code step by step.
"""

OAI_SYS_PROMPT_BACKUP = """
## Instruction

Solve the following math problem efficiently and clearly:
You may use symbolic math tools if necessary.
You do not need to solve everything in one go:
You can use the `sympy_execute` tool to execute sympy code step by step.

- For simple problems (2 steps or fewer):
  Provide a concise solution with minimal explanation.

- For complex problems (3 steps or more):
  Use this step-by-step format:

  ## Step 1: [Concise description]
  [Brief explanation and calculations]

  ## Step 2: [Concise description]
  [Brief explanation and calculations]

  ...

Regardless of the approach, always conclude with:

Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.

Where [answer] is just the final number or expression that solves the problem.


## Example Problem:

Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.

## Step 1: Consider the expression inside the first square root
The expression inside the first square root is $x-2$, and it must be non-negative. This means that $x-2 \\ge 0$, which simplifies to $x \\ge 2$.

## Step 2: Consider the expression inside the second square root
The expression inside the second square root is $5-x$, and it must be non-negative. This means that $5-x \\ge 0$, which simplifies to $x \\le 5$.

## Step 3: Consider the denominator of the expression
The denominator of the expression is $\\sqrt{5-x}$, and it cannot be equal to zero. This means that $5-x>0$, which simplifies to $x<5$.

## Step 4: Combine the results from the previous steps
Combining the results from the previous steps, we have $x \\ge 2$ and $x < 5$. This means that the domain of the expression is $[2,5)$.

The final answer is: $\\boxed{[2,5)}$. I hope it is correct.

"""