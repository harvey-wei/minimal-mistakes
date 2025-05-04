---
title: "Understanding Float32, Float16, and BFloat16 in PyTorch"
categories: [Machine Learning]
# layout: single
# toc: true
# toc_min_header: 1
# toc_max_header: 1
mathjax: true
---

# 1. What Are Floating-Point Formats?

Floating-point numbers follow the IEEE 754 standard, which encodes a number using:

- **Sign bit** (1 bit)
- **Exponent bits** (with a bias)
- **Mantissa** (fraction) bits)

Each format allocates a different number of bits to the exponent and mantissa, which affects range and precision.

- **Exponent bits** determine the dynamic range. More exponent bits allow the format to represent both extremely large and extremely small numbers. These bits are interpreted with a bias to encode both positive and negative powers of 2.
- **Mantissa bits** (also known as the fraction or significand) determine the level of precision. A greater number of mantissa bits enables the representation of numbers with more significant digits, reducing rounding error. In normalized numbers, a leading 1 is assumed and not stored explicitly, improving efficiency.

A normalized floating-point number is represented as:

$$
(-1)^s \times (1 + \text{fraction}) \times 2^{e - \text{bias}}
$$

Where:

- \( s \) is the sign bit (0 for positive, 1 for negative)
- *fraction* is the mantissa bits interpreted as a binary fraction
- \( e \) is the exponent field (unsigned integer)
- *bias* is a format-specific constant to allow negative exponents

> For example, in `float32`, the exponent bias is 127. If the exponent bits store the value 130, the actual exponent is \( 130 - 127 = 3 \). This allows the format to represent both very large and very small values symmetrically around zero.

---

# 2. Bit Layout Comparison

| Format   | Sign | Exponent | Mantissa | Total Bits | Bias |
|----------|------|----------|----------|------------|------|
| float16  | 1    | 5        | 10       | 16         | 15   |
| bfloat16 | 1    | 8        | 7        | 16         | 127  |
| float32  | 1    | 8        | 23       | 32         | 127  |

The intution is that bfloat16 is a truncated version of float32, keeping the exponent bits and
discarding some mantissa bits. This allows bfloat16 to represent a similar range of values as
float32 but with less precision.
---

# 3. Dynamic Range and Precision

### Key Concepts

- **Dynamic Range**: The range from the smallest to the largest representable number.
- **Machine Epsilon**: The smallest distinguishable difference near 1.0, computed as:

$$
\varepsilon = 2^{-t}
$$

Where $ t $ is the number of mantissa bits.

### Summary Table
---

| Format   | Mantissa Bits | $ \varepsilon $ | Max Value        | Min Normal        | Subnormal Min        |
|----------|----------------|--------------------|-------------------|--------------------|------------------------|
| float16  | 10             | $ 2^{-10} \approx 9.77 \times 10^{-4} $ | $ \sim 6.55\times10^4 $    | $ \sim 6.10\times10^{-5}  $ | $ \sim 5.96\times10^{-8} $ |
| bfloat16 | 7              | $ 2^{-7} \approx 7.81 \times 10^{-3} $  | $ \sim 3.39\times10^{38} $ | $ \sim 1.18\times10^{-38} $ | N/A |
| float32  | 23             | $ 2^{-23} \approx 1.19 \times 10^{-7} $ | $ \sim 3.40\times10^{38} $ | $ \sim 1.18\times10^{-38} $| $ \sim 1.40\times10^{-45} $ |

---

# 4. PyTorch: Inspecting Float Info

Use `torch.finfo` to retrieve floating-point properties:

- `max`: Largest representable finite value
- `min`: Smallest positive normalized value
- `tiny`: Smallest subnormal value (if supported)
- `eps`: Machine epsilon (resolution near 1.0)
- `bits`: Total bits

```python
import torch

for dtype in [torch.float16, torch.bfloat16, torch.float32]:
    info = torch.finfo(dtype)
    print(f"\n--- {dtype} ---")
    print(f"Max:        {info.max:.3e}")
    print(f"Min (norm): {info.min:.3e}")
    print(f"Subnormal:  {info.tiny:.3e}")
    print(f"Epsilon:    {info.eps:.3e}")
    print(f"Bits:       {info.bits}")
```
---
# 5. Why Machine Precision Matters
Since computers store floating-point values with limited precision, tiny differences may not be represented exactly. This leads to rounding and comparison errors:
```python
import math

print(0.1 + 0.2 == 0.3)  # False due to precision error
print(math.isclose(0.1 + 0.2, 0.3, rel_tol=1e-9))  # True
```

Use `math.isclose` or `torch.isclose` to compare floats safely:
```python

import torch

a = torch.tensor([0.1 + 0.2], dtype=torch.float32)
b = torch.tensor([0.3], dtype=torch.float32)

print(torch.isclose(a, b, rtol=1e-9))  # tensor([True])
```
---


# 6. Use in Deep Learning

| Format   | Use Case                          | Pros                             | Cons                               |
|----------|-----------------------------------|----------------------------------|------------------------------------|
| float16  | Mixed precision training          | Fast, compact                    | Risk of underflow/overflow         |
| bfloat16 | TPU/GPU training (NVIDIA A100+)   | Same dynamic range as float32    | Lower precision (7-bit mantissa)   |
| float32  | Default for training , inference  | Accurate, stable                 | Higher memory/compute requirements |

# 7.Conclusion

Understanding floating-point formats is key to building efficient, stable deep learning models. Use
`float32` when precision is critical, `bfloat16` for wide-range approximate computation, and
`float16` when maximizing performance under memory constraints. `bfloat16` achieves roughly the same
dynamic range as `float32` but with much lower precision $ 7.81 \times 10^{-3} $.


# References

- [Wikipedia: Single-precision floating-point format](https://en.wikipedia.org/wiki/Single-precision_floating-point_format)
