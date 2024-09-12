"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<6x4xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3xbf16>, tensor<bf16>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<6x4xbf16>
    %5 = "stablehlo.pad"(%3#0, %3#1) <{edge_padding_high = array<i64: 2, 1>, edge_padding_low = array<i64: 1, 0>, interior_padding = array<i64: 1, 0>}> : (tensor<2x3xbf16>, tensor<bf16>) -> tensor<6x4xbf16>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<6x4xbf16>, tensor<6x4xbf16>) -> ()
    "func.return"(%5) : (tensor<6x4xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3xbf16>, tensor<bf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-3.604890e-04, -2.956390e-04, 2.075200e-03], [-8.087160e-04, 1.773830e-04, -3.852840e-04]]> : tensor<2x3xbf16>}> : () -> tensor<2x3xbf16>
    %2 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
    "func.return"(%1, %2) : (tensor<2x3xbf16>, tensor<bf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<6x4xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [-3.604890e-04, -2.956390e-04, 2.075200e-03, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [-8.087160e-04, 1.773830e-04, -3.852840e-04, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00]]> : tensor<6x4xbf16>}> : () -> tensor<6x4xbf16>
    "func.return"(%0) : (tensor<6x4xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

