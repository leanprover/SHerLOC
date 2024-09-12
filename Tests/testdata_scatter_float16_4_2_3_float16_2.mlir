"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[3, 2]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3xf16>, tensor<2xf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3xf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      "stablehlo.return"(%arg1) : (tensor<f16>) -> ()
    }) : (tensor<4x2x3xf16>, tensor<2xi64>, tensor<2xf16>) -> tensor<4x2x3xf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x2x3xf16>, tensor<4x2x3xf16>) -> ()
    "func.return"(%6) : (tensor<4x2x3xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3xf16>, tensor<2xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[6.870120e-01, 2.033200e+00, 2.789060e+00], [1.398440e+00, 7.309570e-01, 2.519530e+00]], [[-3.044920e+00, 2.149660e-01, -5.921880e+00], [-5.277340e+00, 1.605470e+00, 4.816410e+00]], [[-2.134770e+00, -4.492190e-01, -1.436520e+00], [-1.509770e+00, 1.167970e+00, -3.921880e+00]], [[9.851560e+00, 1.700200e+00, -1.511720e+00], [-2.066410e+00, 3.126950e+00, 2.890630e+00]]]> : tensor<4x2x3xf16>}> : () -> tensor<4x2x3xf16>
    %2 = "stablehlo.constant"() <{value = dense<[-5.906250e+00, 2.890630e+00]> : tensor<2xf16>}> : () -> tensor<2xf16>
    "func.return"(%1, %2) : (tensor<4x2x3xf16>, tensor<2xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[6.870120e-01, 2.033200e+00, 2.789060e+00], [1.398440e+00, 7.309570e-01, 2.519530e+00]], [[-3.044920e+00, 2.149660e-01, -5.921880e+00], [-5.277340e+00, 1.605470e+00, 4.816410e+00]], [[-2.134770e+00, -4.492190e-01, -1.436520e+00], [-1.509770e+00, 1.167970e+00, -3.921880e+00]], [[9.851560e+00, 1.700200e+00, -5.906250e+00], [-2.066410e+00, 3.126950e+00, 2.890630e+00]]]> : tensor<4x2x3xf16>}> : () -> tensor<4x2x3xf16>
    "func.return"(%0) : (tensor<4x2x3xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

