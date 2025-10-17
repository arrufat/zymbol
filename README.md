## Zymbol

**Symbolic Automatic Differentiation for Zig**

```zig
const zymbol = @import("zymbol");

var registry = zymbol.OpRegistry.init(allocator);
try registry.registerBuiltins();

// Define function from string
var f = try zymbol.parse(allocator, &registry, "x^2 + sin(x)");
defer f.deinit();

// Take derivatives symbolically
var df = try zymbol.grad(f, "x");
defer df.deinit();

// Compose derivatives
var d2f = try zymbol.grad(df, "x");
defer d2f.deinit();

// Inspect the symbolic form
const df_str = try df.toString();
defer allocator.free(df_str);
std.debug.print("df/dx = {s}\n", .{df_str});
```

### Features:
- **Symbolic differentiation** - generates derivative expressions at compile time
- **Composable** - `grad(grad(f))` for higher-order derivatives
- **Extensible** - register custom operations without modifying the library
- **Inspectable** - convert graphs back to human-readable strings
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
