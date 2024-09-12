"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x2x3xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x2x3xbf16>, tensor<2x3xbf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x2x3xbf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%7) : (tensor<bf16>) -> ()
    }) : (tensor<1x2x3xbf16>, tensor<1xi64>, tensor<2x3xbf16>) -> tensor<1x2x3xbf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<1x2x3xbf16>, tensor<1x2x3xbf16>) -> ()
    "func.return"(%6) : (tensor<1x2x3xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x2x3xbf16>, tensor<2x3xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[4.875000e+00, -2.207030e-01, -6.250000e-02], [-3.484380e+00, 7.625000e+00, -7.812500e-01]]]> : tensor<1x2x3xbf16>}> : () -> tensor<1x2x3xbf16>
    %2 = "stablehlo.constant"() <{value = dense<[[-3.765630e+00, -3.015630e+00, -3.015630e+00], [-2.046880e+00, 1.859380e+00, 2.640630e+00]]> : tensor<2x3xbf16>}> : () -> tensor<2x3xbf16>
    "func.return"(%1, %2) : (tensor<1x2x3xbf16>, tensor<2x3xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x2x3xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[-3.765630e+00, -3.015630e+00, -3.015630e+00], [-3.484380e+00, 1.859380e+00, -7.812500e-01]]]> : tensor<1x2x3xbf16>}> : () -> tensor<1x2x3xbf16>
    "func.return"(%0) : (tensor<1x2x3xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

