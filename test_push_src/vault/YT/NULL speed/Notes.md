---
tags:
  - YT
  - Notes
---
# Idea

Sometimes IDEs will hint you to use NULL != ptr rather than ptr != NULL for speed reasons. But why?

## GPT
  
In C, `NULL` is typically defined as `(void *)0`, which is a constant representing a null pointer. The expressions `NULL != ptr` and `ptr != NULL` are functionally equivalent, and both will be optimized to the same machine code by a good compiler. The difference in performance, if any, is likely to be negligible.

The reason your IDE may suggest `NULL != ptr` as potentially faster is related to a common programming mistake. If you accidentally write `=` instead of `==` in a conditional statement, like this:

```c
if (ptr = NULL) {
	// Code that unintentionally assigns NULL to ptr
}
```


The assignment `ptr = NULL` will always evaluate to true because it assigns `NULL` to `ptr` and then tests the result of the assignment, which is the assigned value. On the other hand, if you write it as `NULL != ptr`, the compiler will catch this mistake and generate an error because you cannot assign a value to a constant.

Modern compilers often include warnings to catch potential mistakes like this, and some developers find it helpful to use the `NULL != ptr` style to avoid accidental assignments in conditions. However, it's essential to note that a good coding practice is to enable compiler warnings and pay attention to them rather than relying on the order of operands in an equality comparison for safety.