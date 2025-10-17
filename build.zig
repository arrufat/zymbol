const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("zymbol", .{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
    });

    const exe = b.addExecutable(.{
        .name = "zymbol",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zymbol", .module = mod },
            },
        }),
    });

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");

    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const mod_tests = b.addTest(.{
        .root_module = mod,
    });

    const run_mod_tests = b.addRunArtifact(mod_tests);

    const exe_tests = b.addTest(.{
        .root_module = exe.root_module,
    });

    const run_exe_tests = b.addRunArtifact(exe_tests);

    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);

    const wasm_target = b.resolveTargetQuery(.{
        .cpu_arch = .wasm32,
        .os_tag = .freestanding,
    });

    const wasm = b.addExecutable(.{
        .name = "zymbol_wasm",
        .root_module = b.createModule(.{
            .root_source_file = b.path("examples/wasm/zymbol_wasm.zig"),
            .target = wasm_target,
            .optimize = .ReleaseSmall,
            .imports = &.{
                .{ .name = "zymbol", .module = mod },
            },
        }),
    });
    wasm.rdynamic = true;
    wasm.entry = .disabled;

    const install_wasm = b.addInstallArtifact(wasm, .{
        .dest_dir = .{ .override = .{ .custom = "wasm" } },
    });

    const install_www = b.addInstallDirectory(.{
        .source_dir = b.path("examples/wasm/public"),
        .install_dir = .{ .custom = "wasm" },
        .install_subdir = "",
    });

    const wasm_step = b.step("wasm-example", "Build WebAssembly derivative playground");
    wasm_step.dependOn(&install_wasm.step);
    wasm_step.dependOn(&install_www.step);
}
