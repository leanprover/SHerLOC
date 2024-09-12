"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x3x2xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<1x2xf32>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<4x3x2xf32>
    %4 = "stablehlo.broadcast_in_dim"(%2) <{broadcast_dimensions = array<i64: 0, 2>}> : (tensor<1x2xf32>) -> tensor<4x3x2xf32>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x3x2xf32>, tensor<4x3x2xf32>) -> ()
    "func.return"(%4) : (tensor<4x3x2xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x2xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0.544339418, 2.04886508]]> : tensor<1x2xf32>}> : () -> tensor<1x2xf32>
    "func.return"(%1) : (tensor<1x2xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x3x2xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[0.544339418, 2.04886508], [0.544339418, 2.04886508], [0.544339418, 2.04886508]], [[0.544339418, 2.04886508], [0.544339418, 2.04886508], [0.544339418, 2.04886508]], [[0.544339418, 2.04886508], [0.544339418, 2.04886508], [0.544339418, 2.04886508]], [[0.544339418, 2.04886508], [0.544339418, 2.04886508], [0.544339418, 2.04886508]]]> : tensor<4x3x2xf32>}> : () -> tensor<4x3x2xf32>
    "func.return"(%0) : (tensor<4x3x2xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

