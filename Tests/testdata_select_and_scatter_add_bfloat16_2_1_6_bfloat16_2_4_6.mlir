"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x4x6xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x1x6xbf16>, tensor<2x4x6xbf16>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2x4x6xbf16>
    %5 = "stablehlo.constant"() <{value = dense<0xFF80> : tensor<bf16>}> : () -> tensor<bf16>
    %6 = "stablehlo.pad"(%3#1, %5) <{edge_padding_high = array<i64: 0, 0, 0>, edge_padding_low = array<i64: 0, 0, 0>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<2x4x6xbf16>, tensor<bf16>) -> tensor<2x4x6xbf16>
    %7 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
    %8 = "stablehlo.select_and_scatter"(%6, %3#0, %7) <{window_dimensions = array<i64: 1, 3, 1>, window_strides = array<i64: 1, 2, 1>}> ({
    ^bb0(%arg2: tensor<bf16>, %arg3: tensor<bf16>):
      %11 = "stablehlo.compare"(%arg2, %arg3) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GE>}> : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      "stablehlo.return"(%11) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %10 = "stablehlo.add"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%10) : (tensor<bf16>) -> ()
    }) : (tensor<2x4x6xbf16>, tensor<2x1x6xbf16>, tensor<bf16>) -> tensor<2x4x6xbf16>
    %9 = "stablehlo.slice"(%8) <{limit_indices = array<i64: 2, 4, 6>, start_indices = array<i64: 0, 0, 0>, strides = array<i64: 1, 1, 1>}> : (tensor<2x4x6xbf16>) -> tensor<2x4x6xbf16>
    "stablehlo.custom_call"(%9, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x4x6xbf16>, tensor<2x4x6xbf16>) -> ()
    "func.return"(%9) : (tensor<2x4x6xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x1x6xbf16>, tensor<2x4x6xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[-4.937500e+00, -1.914060e+00, -1.093750e+00, -1.789060e+00, -5.281250e+00, 2.703130e+00]], [[5.859380e-01, -3.125000e+00, -1.640630e+00, 2.046880e+00, -1.117190e+00, -1.351560e+00]]]> : tensor<2x1x6xbf16>}> : () -> tensor<2x1x6xbf16>
    %2 = "stablehlo.constant"() <{value = dense<[[[-6.500000e+00, 5.500000e+00, -3.491210e-02, 2.421880e+00, 7.148430e-01, 1.890630e+00], [-4.687500e+00, -5.281250e+00, 1.726560e+00, -9.335930e-01, -1.070310e+00, -5.000000e-01], [8.789060e-01, -1.515630e+00, -3.457030e-01, 1.320310e+00, -1.242190e+00, -2.625000e+00], [3.156250e+00, 1.781250e+00, -1.664060e+00, 5.531250e+00, -1.242190e+00, 7.625000e+00]], [[-2.375000e+00, -1.960940e+00, -1.679690e+00, -1.921880e+00, -3.437500e-01, 1.906250e+00], [1.890630e+00, -1.859380e+00, 1.031250e+00, -1.953130e-01, -1.904300e-01, -8.593750e-01], [-3.750000e+00, -1.826170e-01, 5.625000e+00, -1.078130e+00, -2.843750e+00, 9.125000e+00], [-3.466800e-02, -5.531250e+00, -4.687500e+00, -3.859380e+00, 1.546880e+00, -2.765630e+00]]]> : tensor<2x4x6xbf16>}> : () -> tensor<2x4x6xbf16>
    "func.return"(%1, %2) : (tensor<2x1x6xbf16>, tensor<2x4x6xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x4x6xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[0.000000e+00, -1.914060e+00, 0.000000e+00, -1.789060e+00, -5.281250e+00, 2.703130e+00], [0.000000e+00, 0.000000e+00, -1.093750e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [-4.937500e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [5.859380e-01, 0.000000e+00, 0.000000e+00, 2.046880e+00, -1.117190e+00, 0.000000e+00], [0.000000e+00, -3.125000e+00, -1.640630e+00, 0.000000e+00, 0.000000e+00, -1.351560e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]]> : tensor<2x4x6xbf16>}> : () -> tensor<2x4x6xbf16>
    "func.return"(%0) : (tensor<2x4x6xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

