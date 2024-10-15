# Appendix

## Deeper analysis
TODO: view all of them to understand the strengths of using tools

**Right in CoT but not ReAct:**
| Level | Problem Type          | Index |
|-------|-----------------------|-------|
| 5     | Intermediate Algebra  | 2722  |
| 3     | Number Theory         | 3455  |

**Right in ReAct but not CoT:**
| Level | Problem Type          | Index |
|-------|-----------------------|-------|
| 4     | Algebra               | 935   |
| 1     | Counting & Probability| 1325  |
| 3     | Intermediate Algebra  | 2915  |
| 4     | Geometry              | 1695  |
| 5     | Precalculus           | 4996  |
| 4     | Intermediate Algebra  | 2973  |
| 3     | Counting & Probability| 1633  |
| 5     | Counting & Probability| 1422  |
| 3     | Algebra               | 585   |
| 3     | Prealgebra            | 4310  |
| 4     | Intermediate Algebra  | 2486  |
| 4     | Intermediate Algebra  | 3039  |
| 5     | Geometry              | 1834  |


## Considerations when making sympy tool
- sanitization of inputs to prevent imports
- restrict via whilelisted builtins only
- persisting globals across tool calls for the specific problem as LLM assumes that it will continue from previous equation
- disabling parallel function calling (on by default) as agent might issue multiple calls but one math expression depended on the previous so we get errors

## Bugs
- Comparison errors due to formatting, eg Problem 455 this is graded False `y_true='(1,3)', y_pred='(1, 3)'`  (spacing after the comma)
- `from sympy import *` sometimes causes collision when AI uses variable with same name as sympy stuff (eg `S` is a shortcut for sympify but can be used as a variable for a series, `poly` is a function in sympy but AI uses it for a polynomial variable). **Must fix**, quite a number of times AI struggles cos of this!
- default function calling with LangGraph ReAct agent (not sure if its specifically function calling or OAI function call format) causes AI message (and thus prompt for subsequent messages) to contain call ID (hash), so the overall trajectories might not be exactly reproducible (and cant use LM cache!)

## Other
Full old example of fallback to manual calculation

Problem 4236

```
================================ System Message ================================


## Instruction
You will be given a math problem.

Do not solve the problem in one go. Rather, solve it step by step.
Your next reply should either be a new step, or a conclusion that solves the entire problem.

For each step, first write down your thoughts from the previous observation, 
then use the `sympy_execute` tool to execute sympy code.

When you are done, conclude with:

Therefore, the final answer is: $\boxed{answer}$. I hope it is correct.

Where [answer] is just the final number or expression that solves the problem.

================================ Human Message =================================

What is the least perfect square with 3 different prime factors?
================================== Ai Message ==================================

To find the least perfect square with 3 different prime factors, we need to consider the properties of perfect squares and prime factors.

1. A perfect square is a number that can be expressed as the square of an integer.
2. For a number to be a perfect square, all the prime factors in its prime factorization must have even exponents.

Given that we want 3 different prime factors, the smallest perfect square that meets this criterion would be the square of the product of the first three prime numbers.

The first three prime numbers are 2, 3, and 5. 

Next, we will calculate the product of these primes and then square the result. 

Let's calculate this step by step. 

First, I will find the product of the first three prime numbers.
Tool Calls:
  sympy_execute (call_ZreXnNkkfpvvZBDQlYqYnksF)
 Call ID: call_ZreXnNkkfpvvZBDQlYqYnksF
  Args:
    code: from sympy import *
# First three prime numbers
primes = [2, 3, 5]
# Calculate the product
product = 1
for prime in primes:
    product *= prime
product
================================= Tool Message =================================
Name: sympy_execute

Executing code:

# First three prime numbers
primes = [2, 3, 5]
# Calculate the product
product = 1
for prime in primes:
    product *= prime
product

Result: 30
================================== Ai Message ==================================

The product of the first three prime numbers (2, 3, and 5) is 30. 

Now, to find the least perfect square with these three different prime factors, we need to square this product. 

Let's calculate \(30^2\).
Tool Calls:
  sympy_execute (call_C3RvKb6tsLKj5TC081szn4xn)
 Call ID: call_C3RvKb6tsLKj5TC081szn4xn
  Args:
    code: product**2
================================= Tool Message =================================
Name: sympy_execute

Error: unsupported operand type(s) for ** or pow(): 'function' and 'int'
================================== Ai Message ==================================

It seems there was an error in the calculation. I will explicitly define the product as a number and then square it. Let's try that again.
Tool Calls:
  sympy_execute (call_dOvsYEJJjziWv7lDFXrDdGji)
 Call ID: call_dOvsYEJJjziWv7lDFXrDdGji
  Args:
    code: 30**2
================================= Tool Message =================================
Name: sympy_execute

Executing code:
30**2

Result: 900
================================== Ai Message ==================================

The square of the product \(30\) is \(900\). 

Thus, the least perfect square with 3 different prime factors (2, 3, and 5) is \(900\).

Therefore, the final answer is: \(\boxed{900}\). I hope it is correct.
y_true='900', y_pred='900'
Problem: What is the least perfect square with 3 different prime factors?
Model Answer: The square of the product \(30\) is \(900\). 

Thus, the least perfect square with 3 different prime factors (2, 3, and 5) is \(900\).

Therefore, the final answer is: \(\boxed{900}\). I hope it is correct.
```

## Personal learnings
- MATH is a nice and simple env to iterate fast. I previously did a lot more complex envs like USACO-bench, SWE-Bench, WebArena -- and spent a ton of time engineering and not iterating on ideas.
- I need to learn how to use open source (eg Llama) models more
