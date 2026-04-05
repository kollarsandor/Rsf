pub const struct_futhark_context_config = opaque {};
pub const struct_futhark_context = opaque {};
pub const struct_futhark_f32_1d = opaque {};
pub const struct_futhark_f32_2d = opaque {};
pub const struct_futhark_f32_3d = opaque {};
pub const struct_futhark_u64_1d = opaque {};
pub const struct_futhark_i64_1d = opaque {};
pub const struct_futhark_u32_1d = opaque {};
pub const struct_futhark_i32_1d = opaque {};
pub const struct_futhark_bool_1d = opaque {};

pub extern "c" fn futhark_context_config_new() ?*struct_futhark_context_config;
pub extern "c" fn futhark_context_config_free(cfg: ?*struct_futhark_context_config) void;
pub extern "c" fn futhark_context_config_set_device(cfg: ?*struct_futhark_context_config, device: [*:0]const u8) void;
pub extern "c" fn futhark_context_config_set_default_group_size(cfg: ?*struct_futhark_context_config, size: c_int) void;
pub extern "c" fn futhark_context_config_set_default_num_groups(cfg: ?*struct_futhark_context_config, num: c_int) void;
pub extern "c" fn futhark_context_config_set_default_tile_size(cfg: ?*struct_futhark_context_config, size: c_int) void;
pub extern "c" fn futhark_context_config_set_default_thread_block_size(cfg: ?*struct_futhark_context_config, size: c_int) void;
pub extern "c" fn futhark_context_config_set_default_grid_size(cfg: ?*struct_futhark_context_config, size: c_int) void;
pub extern "c" fn futhark_context_config_set_default_threshold(cfg: ?*struct_futhark_context_config, size: c_int) void;

pub extern "c" fn futhark_context_new(cfg: ?*struct_futhark_context_config) ?*struct_futhark_context;
pub extern "c" fn futhark_context_free(ctx: ?*struct_futhark_context) void;
pub extern "c" fn futhark_context_sync(ctx: ?*struct_futhark_context) c_int;
pub extern "c" fn futhark_context_get_error(ctx: ?*struct_futhark_context) ?[*:0]const u8;

pub extern "c" fn futhark_new_f32_1d(ctx: ?*struct_futhark_context, data: ?[*]const f32, dim0: i64) ?*struct_futhark_f32_1d;
pub extern "c" fn futhark_new_f32_2d(ctx: ?*struct_futhark_context, data: ?[*]const f32, dim0: i64, dim1: i64) ?*struct_futhark_f32_2d;
pub extern "c" fn futhark_new_f32_3d(ctx: ?*struct_futhark_context, data: ?[*]const f32, dim0: i64, dim1: i64, dim2: i64) ?*struct_futhark_f32_3d;
pub extern "c" fn futhark_new_i64_1d(ctx: ?*struct_futhark_context, data: ?[*]const i64, dim0: i64) ?*struct_futhark_i64_1d;
pub extern "c" fn futhark_new_u64_1d(ctx: ?*struct_futhark_context, data: ?[*]const u64, dim0: i64) ?*struct_futhark_u64_1d;

pub extern "c" fn futhark_free_f32_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_1d) c_int;
pub extern "c" fn futhark_free_f32_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_2d) c_int;
pub extern "c" fn futhark_free_f32_3d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_3d) c_int;
pub extern "c" fn futhark_free_i64_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_i64_1d) c_int;
pub extern "c" fn futhark_free_u64_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_u64_1d) c_int;

pub extern "c" fn futhark_values_f32_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_1d, data: ?[*]f32) c_int;
pub extern "c" fn futhark_values_f32_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_2d, data: ?[*]f32) c_int;
pub extern "c" fn futhark_values_f32_3d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_3d, data: ?[*]f32) c_int;
pub extern "c" fn futhark_values_i64_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_i64_1d, data: ?[*]i64) c_int;
pub extern "c" fn futhark_values_u64_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_u64_1d, data: ?[*]u64) c_int;

pub extern "c" fn futhark_shape_f32_1d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_1d) ?[*]const i64;
pub extern "c" fn futhark_shape_f32_2d(ctx: ?*struct_futhark_context, arr: ?*struct_futhark_f32_2d) ?[*]const i64;

pub extern "c" fn futhark_entry_matmul(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_2d, a: ?*const struct_futhark_f32_2d, b: ?*const struct_futhark_f32_2d) c_int;
pub extern "c" fn futhark_entry_batch_matmul(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_3d, a: ?*const struct_futhark_f32_3d, b: ?*const struct_futhark_f32_3d) c_int;
pub extern "c" fn futhark_entry_dot(ctx: ?*struct_futhark_context, out: ?*f32, a: ?*const struct_futhark_f32_1d, b: ?*const struct_futhark_f32_1d) c_int;
pub extern "c" fn futhark_entry_apply_softmax(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_1d, x: ?*const struct_futhark_f32_1d) c_int;
pub extern "c" fn futhark_entry_apply_layer_norm(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_1d, x: ?*const struct_futhark_f32_1d, gamma: ?*const struct_futhark_f32_1d, beta: ?*const struct_futhark_f32_1d, eps: f32) c_int;
pub extern "c" fn futhark_entry_apply_relu(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_1d, x: ?*const struct_futhark_f32_1d) c_int;
pub extern "c" fn futhark_entry_apply_gelu(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_1d, x: ?*const struct_futhark_f32_1d) c_int;
pub extern "c" fn futhark_entry_clip_fisher(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_1d, fisher: ?*const struct_futhark_f32_1d, clip_val: f32) c_int;
pub extern "c" fn futhark_entry_reduce_gradients(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_1d, gradients: ?*const struct_futhark_f32_2d) c_int;
pub extern "c" fn futhark_entry_rank_segments(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_1d, query_hash: u64, segment_hashes: ?*const struct_futhark_u64_1d, base_scores: ?*const struct_futhark_f32_1d) c_int;

pub extern "c" fn futhark_entry_rsf_forward(
    ctx: ?*struct_futhark_context,
    out0: ?*?*struct_futhark_f32_1d,
    in0: ?*const struct_futhark_f32_1d,
    in1: ?*const struct_futhark_f32_1d,
    in2: ?*const struct_futhark_f32_1d,
    in3: ?*const struct_futhark_f32_1d,
    in4: ?*const struct_futhark_f32_1d,
    in5: ?*const struct_futhark_i64_1d,
) c_int;

pub extern "c" fn futhark_entry_rsf_backward(
    ctx: ?*struct_futhark_context,
    out0: ?*?*struct_futhark_f32_1d,
    out1: ?*?*struct_futhark_f32_1d,
    out2: ?*?*struct_futhark_f32_1d,
    out3: ?*?*struct_futhark_f32_1d,
    out4: ?*?*struct_futhark_f32_1d,
    in0: ?*const struct_futhark_f32_1d,
    in1: ?*const struct_futhark_f32_1d,
    in2: ?*const struct_futhark_f32_1d,
    in3: ?*const struct_futhark_f32_1d,
    in4: ?*const struct_futhark_i64_1d,
) c_int;

pub extern "c" fn futhark_entry_compute_rsf_context(
    ctx: ?*struct_futhark_context,
    out0: ?*?*struct_futhark_f32_2d,
    in0: ?*const struct_futhark_f32_2d,
    in1: ?*const struct_futhark_f32_2d,
    in2: ?*const struct_futhark_f32_2d,
    in3: ?*const struct_futhark_f32_1d,
    in4: ?*const struct_futhark_f32_1d,
    in5: f32,
) c_int;

pub extern "c" fn futhark_entry_compute_natural_grad(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_1d, grad: ?*const struct_futhark_f32_1d, fisher: ?*const struct_futhark_f32_1d, damping: f32) c_int;
pub extern "c" fn futhark_entry_update_fisher(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_1d, old_fisher: ?*const struct_futhark_f32_1d, grad: ?*const struct_futhark_f32_1d, decay: f32) c_int;

pub extern "c" fn futhark_entry_add_arrays(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_1d, a: ?*const struct_futhark_f32_1d, b: ?*const struct_futhark_f32_1d) c_int;
pub extern "c" fn futhark_entry_sub_arrays(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_1d, a: ?*const struct_futhark_f32_1d, b: ?*const struct_futhark_f32_1d) c_int;
pub extern "c" fn futhark_entry_mul_arrays(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_1d, a: ?*const struct_futhark_f32_1d, b: ?*const struct_futhark_f32_1d) c_int;
pub extern "c" fn futhark_entry_div_arrays(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_1d, a: ?*const struct_futhark_f32_1d, b: ?*const struct_futhark_f32_1d) c_int;
pub extern "c" fn futhark_entry_mul_scalar(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_1d, a: ?*const struct_futhark_f32_1d, s: f32) c_int;
pub extern "c" fn futhark_entry_add_scalar(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_1d, a: ?*const struct_futhark_f32_1d, s: f32) c_int;
pub extern "c" fn futhark_entry_div_scalar(ctx: ?*struct_futhark_context, out: ?*?*struct_futhark_f32_1d, a: ?*const struct_futhark_f32_1d, s: f32) c_int;

pub extern "c" fn futhark_entry_array_sum(ctx: ?*struct_futhark_context, out: ?*f32, a: ?*const struct_futhark_f32_1d) c_int;
pub extern "c" fn futhark_entry_array_mean(ctx: ?*struct_futhark_context, out: ?*f32, a: ?*const struct_futhark_f32_1d) c_int;
pub extern "c" fn futhark_entry_array_max(ctx: ?*struct_futhark_context, out: ?*f32, a: ?*const struct_futhark_f32_1d) c_int;
pub extern "c" fn futhark_entry_array_min(ctx: ?*struct_futhark_context, out: ?*f32, a: ?*const struct_futhark_f32_1d) c_int;
