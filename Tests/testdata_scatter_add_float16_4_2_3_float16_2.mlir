"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[3, 2]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3xf16>, tensor<2xf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3xf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
      "stablehlo.return"(%7) : (tensor<f16>) -> ()
    }) : (tensor<4x2x3xf16>, tensor<2xi64>, tensor<2xf16>) -> tensor<4x2x3xf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x2x3xf16>, tensor<4x2x3xf16>) -> ()
    "func.return"(%6) : (tensor<4x2x3xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3xf16>, tensor<2xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[-1.169920e+00, 4.914550e-01, 4.372560e-01], [-5.963130e-02, -2.089840e+00, 2.812500e+00]], [[2.240230e+00, -2.693360e+00, 2.982420e+00], [-2.194820e-01, 1.992190e+00, 1.350590e+00]], [[1.885740e+00, -1.938480e+00, 1.156250e+00], [8.725580e-01, -1.982420e+00, 5.625000e+00]], [[1.516600e+00, -3.257810e+00, -2.289060e+00], [7.890630e-01, -1.298830e+00, -1.415040e+00]]]> : tensor<4x2x3xf16>}> : () -> tensor<4x2x3xf16>
    %2 = "stablehlo.constant"() <{value = dense<[-5.737300e-02, 7.153320e-01]> : tensor<2xf16>}> : () -> tensor<2xf16>
    "func.return"(%1, %2) : (tensor<4x2x3xf16>, tensor<2xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[-1.169920e+00, 4.914550e-01, 4.372560e-01], [-5.963130e-02, -2.089840e+00, 2.812500e+00]], [[2.240230e+00, -2.693360e+00, 2.982420e+00], [-2.194820e-01, 1.992190e+00, 1.350590e+00]], [[1.885740e+00, -1.938480e+00, 1.156250e+00], [8.725580e-01, -1.982420e+00, 5.625000e+00]], [[1.516600e+00, -3.257810e+00, -2.345700e+00], [7.890630e-01, -1.298830e+00, -6.997070e-01]]]> : tensor<4x2x3xf16>}> : () -> tensor<4x2x3xf16>
    "func.return"(%0) : (tensor<4x2x3xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

