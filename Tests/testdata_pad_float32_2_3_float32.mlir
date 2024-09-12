"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x0xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3xf32>, tensor<f32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2x0xf32>
    %5 = "stablehlo.pad"(%3#0, %3#1) <{edge_padding_high = array<i64: 0, -3>, edge_padding_low = array<i64: 0, -2>, interior_padding = array<i64: 0, 1>}> : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x0xf32>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x0xf32>, tensor<2x0xf32>) -> ()
    "func.return"(%5) : (tensor<2x0xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3xf32>, tensor<f32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[2.38316075E-4, -1.74506466E-4, 0.00169209205], [3.11927375E-4, 0.00166800153, 6.94549758E-4]]> : tensor<2x3xf32>}> : () -> tensor<2x3xf32>
    %2 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    "func.return"(%1, %2) : (tensor<2x3xf32>, tensor<f32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x0xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<> : tensor<2x0xf32>}> : () -> tensor<2x0xf32>
    "func.return"(%0) : (tensor<2x0xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

