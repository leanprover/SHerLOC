"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x2x3xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x2x3xf16>, tensor<2x3xf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x2x3xf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      "stablehlo.return"(%arg1) : (tensor<f16>) -> ()
    }) : (tensor<1x2x3xf16>, tensor<1xi64>, tensor<2x3xf16>) -> tensor<1x2x3xf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<1x2x3xf16>, tensor<1x2x3xf16>) -> ()
    "func.return"(%6) : (tensor<1x2x3xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x2x3xf16>, tensor<2x3xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[2.154300e+00, 4.125000e+00, -5.961910e-01], [-2.064450e+00, 4.726560e+00, 5.160160e+00]]]> : tensor<1x2x3xf16>}> : () -> tensor<1x2x3xf16>
    %2 = "stablehlo.constant"() <{value = dense<[[-2.859380e+00, 1.833010e+00, -9.399410e-01], [-1.495120e+00, -1.185550e+00, 5.166020e-01]]> : tensor<2x3xf16>}> : () -> tensor<2x3xf16>
    "func.return"(%1, %2) : (tensor<1x2x3xf16>, tensor<2x3xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x2x3xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[-2.859380e+00, 1.833010e+00, -9.399410e-01], [-1.495120e+00, -1.185550e+00, 5.166020e-01]]]> : tensor<1x2x3xf16>}> : () -> tensor<1x2x3xf16>
    "func.return"(%0) : (tensor<1x2x3xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

