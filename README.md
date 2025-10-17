## Zymbol

**Symbolic Automatic Differentiation for Zig**

```zig
const std = @import("std");
const zymbol = @import("zymbol");

var registry = zymbol.Registry.init(allocator);
defer registry.deinit();
try registry.registerBuiltins();

var expr = try zymbol.Expression.parse(allocator, &registry, "x^2 + sin(x)");
defer expr.deinit();

var inputs = std.StringHashMap(f32).init(allocator);
defer inputs.deinit();
try inputs.put("x", 1.5);

const value = try expr.evaluate(inputs);
std.debug.print("f(1.5) = {d}\n", .{value});

var grad_expr = try expr.symbolicGradient("x");
defer grad_expr.deinit();

const grad_value = try grad_expr.evaluate(inputs);
std.debug.print("df/dx(1.5) = {d}\n", .{grad_value});
```

### Features:
- **Symbolic differentiation** - generates derivative expressions at compile time
- **Composable** - `grad(grad(f))` for higher-order derivatives
- **Extensible** - register custom operations without modifying the library
- **Inspectable** - convert graphs back to human-readable strings
- **Simplified output** - algebraic pass folds constants, collapses duplicate terms, and removes trivial factors (e.g. `d/dx x^3` -> `3 * (x ^ 2)`, `x + x` -> `2 * x`)
- **Zero runtime overhead** - all parsing happens at compile time (when used with comptime)
- **Robust gradients** - gracefully handles edge cases like non-positive power bases to avoid NaNs

### Expression Syntax
- Identifiers map to input variables (e.g. `x`, `foo_bar`).
- Literals support integers and floating-point values (`2`, `3.14`).
- Binary operators: `+`, `-`, `*`, `/`, `^` with standard precedence.
- Unary prefixes `+` and `-` are supported (`-x`, `-(x + 1)`).
- Parentheses group sub-expressions.
- Function calls invoke registered operations (`sin(x)`, `max(x, y)`, etc.).

### WebAssembly Playground
Build an interactive derivative playground that runs entirely in the browser:

```bash
zig build wasm-example
cd zig-out/wasm
python -m http.server 8080
```

Then browse to <http://localhost:8080>. The page lets you type an expression, choose the differentiation variable, and calls the compiled WASM module to print the symbolic derivative. The build enables `rdynamic` so every exported Zig function stays visible to JavaScript.

Changes pushed to `master` are deployed automatically to <https://arrufat.github.io/zymbol> via the GitHub Actions workflow in `.github/workflows/pages.yml`, which builds with the latest Zig master toolchain.
