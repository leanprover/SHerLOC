"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x4x6xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x3x5xbf16>, tensor<2x4x6xbf16>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2x4x6xbf16>
    %5 = "stablehlo.constant"() <{value = dense<0xFF80> : tensor<bf16>}> : () -> tensor<bf16>
    %6 = "stablehlo.pad"(%3#1, %5) <{edge_padding_high = array<i64: 0, 0, 0>, edge_padding_low = array<i64: 0, 0, 0>, interior_padding = array<i64: 0, 0, 0>}> : (tensor<2x4x6xbf16>, tensor<bf16>) -> tensor<2x4x6xbf16>
    %7 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
    %8 = "stablehlo.select_and_scatter"(%6, %3#0, %7) <{window_dimensions = array<i64: 2, 2, 2>}> ({
    ^bb0(%arg2: tensor<bf16>, %arg3: tensor<bf16>):
      %11 = "stablehlo.compare"(%arg2, %arg3) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GE>}> : (tensor<bf16>, tensor<bf16>) -> tensor<i1>
      "stablehlo.return"(%11) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %10 = "stablehlo.add"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%10) : (tensor<bf16>) -> ()
    }) : (tensor<2x4x6xbf16>, tensor<1x3x5xbf16>, tensor<bf16>) -> tensor<2x4x6xbf16>
    %9 = "stablehlo.slice"(%8) <{limit_indices = array<i64: 2, 4, 6>, start_indices = array<i64: 0, 0, 0>, strides = array<i64: 1, 1, 1>}> : (tensor<2x4x6xbf16>) -> tensor<2x4x6xbf16>
    "stablehlo.custom_call"(%9, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x4x6xbf16>, tensor<2x4x6xbf16>) -> ()
    "func.return"(%9) : (tensor<2x4x6xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x3x5xbf16>, tensor<2x4x6xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[-2.140630e+00, 1.750000e+00, -6.562500e-01, -1.460940e+00, 3.437500e+00], [2.359380e+00, 1.835940e+00, 4.941410e-01, -1.085940e+00, -7.750000e+00], [-4.218750e+00, 1.968750e+00, -2.171880e+00, -1.945310e+00, 2.890630e-01]]]> : tensor<1x3x5xbf16>}> : () -> tensor<1x3x5xbf16>
    %2 = "stablehlo.constant"() <{value = dense<[[[1.591800e-01, -4.218750e+00, 2.484380e+00, -5.093750e+00, 3.203130e+00, 2.250000e+00], [1.226560e+00, -2.617190e-01, -1.664060e+00, -3.222660e-01, -1.554690e+00, -1.507810e+00], [3.968750e+00, -6.562500e-01, 6.523440e-01, 1.445310e-01, 1.937500e+00, -3.484380e+00], [-5.093750e+00, 6.406250e-01, -8.281250e-01, 3.328130e+00, 4.812500e+00, 3.062500e+00]], [[4.277340e-01, 8.359380e-01, -2.656250e+00, 1.726560e+00, 1.054690e+00, -1.914060e+00], [6.523440e-01, 3.656250e+00, -1.445310e+00, -1.164060e+00, -7.375000e+00, 3.656250e+00], [7.156250e+00, 2.546880e+00, 3.593750e+00, 4.343750e+00, -1.554690e+00, -8.945310e-01], [3.031250e+00, -2.125000e+00, -8.437500e-01, 3.093750e+00, -8.046880e-01, 1.226560e+00]]]> : tensor<2x4x6xbf16>}> : () -> tensor<2x4x6xbf16>
    "func.return"(%1, %2) : (tensor<1x3x5xbf16>, tensor<2x4x6xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x4x6xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[0.000000e+00, 0.000000e+00, -6.562500e-01, 0.000000e+00, -1.460940e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, -1.656250e+00, 0.000000e+00]], [[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 1.445310e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, -4.312500e+00], [-1.859380e+00, 0.000000e+00, 1.968750e+00, -2.765630e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]]> : tensor<2x4x6xbf16>}> : () -> tensor<2x4x6xbf16>
    "func.return"(%0) : (tensor<2x4x6xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

