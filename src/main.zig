const std = @import("std");

const zymbol = @import("root.zig");
const OpRegistry = zymbol.OpRegistry;
const parse = zymbol.parse;
const grad = zymbol.grad;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create and setup registry
    var registry = OpRegistry.init(allocator);
    defer registry.deinit();
    try registry.registerBuiltins();

    // Example 1: ReLU is now just another operation!
    std.debug.print("Example 1: f(x) = relu(x)\n", .{});
    {
        var f = try parse(allocator, &registry, "relu(x)");
        defer f.deinit();

        const f_str = try f.toString();
        defer allocator.free(f_str);
        std.debug.print("  f(x) = {s}\n", .{f_str});

        var inputs = std.StringHashMap(f32).init(allocator);
        defer inputs.deinit();

        try inputs.put("x", 2.0);
        std.debug.print("  f(2) = {d}\n", .{try f.eval(inputs)});

        try inputs.put("x", -1.0);
        std.debug.print("  f(-1) = {d}\n", .{try f.eval(inputs)});

        var df = try grad(&f, "x");
        defer df.deinit();

        const df_str = try df.toString();
        defer allocator.free(df_str);
        std.debug.print("  df/dx = {s}\n", .{df_str});

        try inputs.put("x", 2.0);
        std.debug.print("  df/dx(2) = {d} (expected: 1)\n", .{try df.eval(inputs)});

        try inputs.put("x", -1.0);
        std.debug.print("  df/dx(-1) = {d} (expected: 0)\n\n", .{try df.eval(inputs)});
    }

    // Example 2: Sigmoid
    std.debug.print("Example 2: f(x) = sigmoid(x)\n", .{});
    {
        var f = try parse(allocator, &registry, "sigmoid(x)");
        defer f.deinit();

        const f_str = try f.toString();
        defer allocator.free(f_str);
        std.debug.print("  f(x) = {s}\n", .{f_str});

        var inputs = std.StringHashMap(f32).init(allocator);
        defer inputs.deinit();
        try inputs.put("x", 0.0);

        std.debug.print("  f(0) = {d} (expected: 0.5)\n", .{try f.eval(inputs)});

        var df = try grad(&f, "x");
        defer df.deinit();

        std.debug.print("  df/dx(0) = {d} (expected: 0.25)\n\n", .{try df.eval(inputs)});
    }

    // Example 3: Combining operations
    std.debug.print("Example 3: f(x) = relu(x) * sigmoid(x)\n", .{});
    {
        var f = try parse(allocator, &registry, "relu(x) * sigmoid(x)");
        defer f.deinit();

        const f_str = try f.toString();
        defer allocator.free(f_str);
        std.debug.print("  f(x) = {s}\n", .{f_str});

        var inputs = std.StringHashMap(f32).init(allocator);
        defer inputs.deinit();
        try inputs.put("x", 2.0);

        std.debug.print("  f(2) = {d}\n", .{try f.eval(inputs)});

        const grad_val = try f.gradient("x", inputs);
        std.debug.print("  df/dx(2) = {d}\n\n", .{grad_val});
    }

    // Example 4: Binary custom op - max
    std.debug.print("Example 4: f(x, y) = max(x, y)\n", .{});
    {
        var f = try parse(allocator, &registry, "max(x, y)");
        defer f.deinit();

        const f_str = try f.toString();
        defer allocator.free(f_str);
        std.debug.print("  f(x, y) = {s}\n", .{f_str});

        var inputs = std.StringHashMap(f32).init(allocator);
        defer inputs.deinit();
        try inputs.put("x", 3.0);
        try inputs.put("y", 5.0);

        std.debug.print("  f(3, 5) = {d}\n", .{try f.eval(inputs)});
        // Note: max() lacks toStringGrad, so symbolic grad fails
        // But numeric gradient() still works via backward()
        std.debug.print("  df/dx(3, 5) = {d} (expected: 0)\n", .{try f.gradient("x", inputs)});
        std.debug.print("  df/dy(3, 5) = {d} (expected: 1)\n\n", .{try f.gradient("y", inputs)});
    }

    // Example 5: Power gradients - gradient w.r.t. base
    std.debug.print("Example 5: f(x) = x^2\n", .{});
    {
        var f = try parse(allocator, &registry, "x ^ 2");
        defer f.deinit();

        const f_str = try f.toString();
        defer allocator.free(f_str);
        std.debug.print("  f(x) = {s}\n", .{f_str});

        var df = try grad(&f, "x");
        defer df.deinit();

        const df_str = try df.toString();
        defer allocator.free(df_str);
        std.debug.print("  df/dx = {s}\n", .{df_str});

        var inputs = std.StringHashMap(f32).init(allocator);
        defer inputs.deinit();
        try inputs.put("x", 3.0);

        std.debug.print("  f(3) = {d} (expected: 9)\n", .{try f.eval(inputs)});
        std.debug.print("  df/dx(3) = {d} (expected: 6)\n\n", .{try df.eval(inputs)});
    }

    // Example 6: Power gradients - gradient w.r.t. exponent
    std.debug.print("Example 6: f(x) = 2^x\n", .{});
    {
        var f = try parse(allocator, &registry, "2 ^ x");
        defer f.deinit();

        const f_str = try f.toString();
        defer allocator.free(f_str);
        std.debug.print("  f(x) = {s}\n", .{f_str});

        var df = try grad(&f, "x");
        defer df.deinit();

        const df_str = try df.toString();
        defer allocator.free(df_str);
        std.debug.print("  df/dx = {s}\n", .{df_str});

        var inputs = std.StringHashMap(f32).init(allocator);
        defer inputs.deinit();
        try inputs.put("x", 3.0);

        std.debug.print("  f(3) = {d} (expected: 8)\n", .{try f.eval(inputs)});
        // 2^3 * ln(2) = 8 * 0.693... = 5.545...
        std.debug.print("  df/dx(3) = {d} (expected: ~5.545)\n\n", .{try df.eval(inputs)});
    }

    // Example 7: Power gradients - both variables
    std.debug.print("Example 7: f(x, y) = x^y\n", .{});
    {
        var f = try parse(allocator, &registry, "x ^ y");
        defer f.deinit();

        const f_str = try f.toString();
        defer allocator.free(f_str);
        std.debug.print("  f(x, y) = {s}\n", .{f_str});

        var inputs = std.StringHashMap(f32).init(allocator);
        defer inputs.deinit();
        try inputs.put("x", 3.0);
        try inputs.put("y", 2.0);

        std.debug.print("  f(3, 2) = {d} (expected: 9)\n", .{try f.eval(inputs)});

        var df_dx = try grad(&f, "x");
        defer df_dx.deinit();
        const df_dx_str = try df_dx.toString();
        defer allocator.free(df_dx_str);
        std.debug.print("  df/dx = {s}\n", .{df_dx_str});
        // y * x^(y-1) = 2 * 3^1 = 6
        std.debug.print("  df/dx(3, 2) = {d} (expected: 6)\n", .{try df_dx.eval(inputs)});

        var df_dy = try grad(&f, "y");
        defer df_dy.deinit();
        const df_dy_str = try df_dy.toString();
        defer allocator.free(df_dy_str);
        std.debug.print("  df/dy = {s}\n", .{df_dy_str});
        // x^y * ln(x) = 9 * ln(3) = 9 * 1.099... = 9.887...
        std.debug.print("  df/dy(3, 2) = {d} (expected: ~9.887)\n\n", .{try df_dy.eval(inputs)});
    }

    // Example 8: Demonstrate symbolic grad error for ops without toStringGrad
    std.debug.print("Example 8: Symbolic gradient limitation\n", .{});
    {
        var f = try parse(allocator, &registry, "max(x, y)");
        defer f.deinit();

        std.debug.print("  Attempting grad(max(x, y), 'x')...\n", .{});
        if (grad(&f, "x")) |_| {
            std.debug.print("  ERROR: Should have failed!\n", .{});
        } else |err| {
            std.debug.print("  Expected error: {s}\n", .{@errorName(err)});
            std.debug.print("  Note: Use f.gradient() for numeric gradients instead.\n", .{});
        }
    }
}
