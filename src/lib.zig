const std = @import("std");

pub const types = @import("types.zig");
pub const NodeId = types.NodeId;
pub const Error = types.Error;

pub const operation = @import("operation.zig");
pub const Operation = operation.Operation;
pub const SymbolicOpInfo = operation.SymbolicOpInfo;
pub const SymbolicRuleFn = operation.SymbolicRuleFn;

pub const Registry = @import("registry.zig").Registry;
pub const Expression = @import("expression.zig").Expression;

pub const registerBuiltins = @import("symbolic.zig").registerBuiltins;

test "basic arithmetic" {
    const allocator = std.testing.allocator;
    var registry: Registry = Registry.init(allocator);
    defer registry.deinit();
    try registerBuiltins(&registry);

    var expr: Expression = try Expression.parse(allocator, &registry, "x + y * 2");
    defer expr.deinit();

    var inputs: std.StringHashMap(f32) = std.StringHashMap(f32).init(allocator);
    defer inputs.deinit();
    try inputs.put("x", 1.0);
    try inputs.put("y", 3.0);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), try expr.evaluate(inputs), 0.0001);
}

test "symbolic gradient" {
    const allocator = std.testing.allocator;
    var registry: Registry = Registry.init(allocator);
    defer registry.deinit();
    try registerBuiltins(&registry);

    var expr: Expression = try Expression.parse(allocator, &registry, "x ^ 2");
    defer expr.deinit();

    var grad_expr: Expression = try expr.symbolicGradient("x");
    defer grad_expr.deinit();

    var inputs: std.StringHashMap(f32) = std.StringHashMap(f32).init(allocator);
    defer inputs.deinit();
    try inputs.put("x", 3.0);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), try grad_expr.evaluate(inputs), 0.0001);
}

test "numeric gradient matches" {
    const allocator = std.testing.allocator;
    var registry: Registry = Registry.init(allocator);
    defer registry.deinit();
    try registerBuiltins(&registry);

    var expr: Expression = try Expression.parse(allocator, &registry, "relu(x) * sigmoid(x)");
    defer expr.deinit();

    var inputs: std.StringHashMap(f32) = std.StringHashMap(f32).init(allocator);
    defer inputs.deinit();
    try inputs.put("x", 2.0);

    const numeric = try expr.numericGradient("x", inputs);
    var grad_expr: Expression = try expr.symbolicGradient("x");
    defer grad_expr.deinit();
    const symbolic_grad = try grad_expr.evaluate(inputs);
    try std.testing.expectApproxEqAbs(symbolic_grad, numeric, 0.0001);
}

test "pow simplification" {
    const allocator = std.testing.allocator;
    var registry: Registry = Registry.init(allocator);
    defer registry.deinit();
    try registerBuiltins(&registry);

    var expr: Expression = try Expression.parse(allocator, &registry, "x ^ 3");
    defer expr.deinit();

    var grad_expr: Expression = try expr.symbolicGradient("x");
    defer grad_expr.deinit();

    const grad_str = try grad_expr.toString();
    defer allocator.free(grad_str);

    try std.testing.expect(std.mem.eql(u8, grad_str, "(3 * (x ^ 2))"));
}

test "raw gradient keeps duplicated terms" {
    const allocator = std.testing.allocator;
    var registry: Registry = Registry.init(allocator);
    defer registry.deinit();
    try registerBuiltins(&registry);

    var expr: Expression = try Expression.parse(allocator, &registry, "x * x");
    defer expr.deinit();

    var grad_expr: Expression = try expr.symbolicGradientRaw("x");
    defer grad_expr.deinit();

    const grad_str = try grad_expr.toString();
    defer allocator.free(grad_str);

    try std.testing.expect(std.mem.eql(u8, grad_str, "((1 * x) + (1 * x))"));
}

test "product rule combines like terms" {
    const allocator = std.testing.allocator;
    var registry: Registry = Registry.init(allocator);
    defer registry.deinit();
    try registerBuiltins(&registry);

    var expr: Expression = try Expression.parse(allocator, &registry, "x * x");
    defer expr.deinit();

    var grad_expr: Expression = try expr.symbolicGradient("x");
    defer grad_expr.deinit();

    const grad_str = try grad_expr.toString();
    defer allocator.free(grad_str);

    try std.testing.expect(std.mem.eql(u8, grad_str, "(2 * x)"));
}

test "division simplifies identical terms" {
    const allocator = std.testing.allocator;
    var registry: Registry = Registry.init(allocator);
    defer registry.deinit();
    try registerBuiltins(&registry);

    var expr: Expression = try Expression.parse(allocator, &registry, "x / x");
    defer expr.deinit();

    var simplified: Expression = try expr.simplify();
    defer simplified.deinit();

    const simplified_str = try simplified.toString();
    defer allocator.free(simplified_str);

    try std.testing.expect(std.mem.eql(u8, simplified_str, "1"));
}

test "division recognizes reciprocal" {
    const allocator = std.testing.allocator;
    var registry: Registry = Registry.init(allocator);
    defer registry.deinit();
    try registerBuiltins(&registry);

    var expr: Expression = try Expression.parse(allocator, &registry, "x / (1 / y)");
    defer expr.deinit();

    var simplified: Expression = try expr.simplify();
    defer simplified.deinit();

    const simplified_str = try simplified.toString();
    defer allocator.free(simplified_str);

    try std.testing.expect(std.mem.eql(u8, simplified_str, "(x * y)"));
}

test "tan gradient matches numeric" {
    const allocator = std.testing.allocator;
    var registry: Registry = Registry.init(allocator);
    defer registry.deinit();
    try registerBuiltins(&registry);

    var expr: Expression = try Expression.parse(allocator, &registry, "tan(x)");
    defer expr.deinit();

    var inputs: std.StringHashMap(f32) = std.StringHashMap(f32).init(allocator);
    defer inputs.deinit();
    const angle: f32 = 0.5;
    try inputs.put("x", angle);

    const value = try expr.evaluate(inputs);
    try std.testing.expectApproxEqAbs(@tan(angle), value, 0.0001);

    const numeric = try expr.numericGradient("x", inputs);
    var grad_expr: Expression = try expr.symbolicGradient("x");
    defer grad_expr.deinit();
    const symbolic = try grad_expr.evaluate(inputs);
    try std.testing.expectApproxEqAbs(symbolic, numeric, 0.0001);
}

test "hyperbolic gradients match numeric" {
    const allocator = std.testing.allocator;
    var registry: Registry = Registry.init(allocator);
    defer registry.deinit();
    try registerBuiltins(&registry);

    var expr: Expression = try Expression.parse(allocator, &registry, "sinh(x) + cosh(x) + tanh(x)");
    defer expr.deinit();

    var inputs: std.StringHashMap(f32) = std.StringHashMap(f32).init(allocator);
    defer inputs.deinit();
    const value: f32 = 0.3;
    try inputs.put("x", value);

    const numeric = try expr.numericGradient("x", inputs);
    var grad_expr: Expression = try expr.symbolicGradient("x");
    defer grad_expr.deinit();
    const symbolic = try grad_expr.evaluate(inputs);
    try std.testing.expectApproxEqAbs(symbolic, numeric, 0.0001);

    const expected_val = std.math.sinh(value) + std.math.cosh(value) + std.math.tanh(value);
    try std.testing.expectApproxEqAbs(expected_val, try expr.evaluate(inputs), 0.0001);
}
