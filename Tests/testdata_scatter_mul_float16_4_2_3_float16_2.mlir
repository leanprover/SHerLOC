"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[3, 2]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3xf16>, tensor<2xf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3xf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
      "stablehlo.return"(%7) : (tensor<f16>) -> ()
    }) : (tensor<4x2x3xf16>, tensor<2xi64>, tensor<2xf16>) -> tensor<4x2x3xf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x2x3xf16>, tensor<4x2x3xf16>) -> ()
    "func.return"(%6) : (tensor<4x2x3xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3xf16>, tensor<2xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[-3.472660e+00, -2.609380e+00, -4.694820e-01], [-3.058590e+00, -6.464840e-01, -2.281250e+00]], [[-1.517580e+00, 1.289060e+00, 2.335940e+00], [-5.253910e+00, 3.238280e+00, -7.416990e-01]], [[-3.402340e+00, -5.883790e-01, -2.954100e-01], [2.277340e+00, 1.465820e+00, 4.523440e+00]], [[-2.695310e-01, 2.501950e+00, 2.963870e-01], [-4.585940e+00, 2.514650e-01, -1.791020e+00]]]> : tensor<4x2x3xf16>}> : () -> tensor<4x2x3xf16>
    %2 = "stablehlo.constant"() <{value = dense<[-2.777340e+00, -1.203130e+00]> : tensor<2xf16>}> : () -> tensor<2xf16>
    "func.return"(%1, %2) : (tensor<4x2x3xf16>, tensor<2xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[-3.472660e+00, -2.609380e+00, -4.694820e-01], [-3.058590e+00, -6.464840e-01, -2.281250e+00]], [[-1.517580e+00, 1.289060e+00, 2.335940e+00], [-5.253910e+00, 3.238280e+00, -7.416990e-01]], [[-3.402340e+00, -5.883790e-01, -2.954100e-01], [2.277340e+00, 1.465820e+00, 4.523440e+00]], [[-2.695310e-01, 2.501950e+00, -8.232420e-01], [-4.585940e+00, 2.514650e-01, 2.154300e+00]]]> : tensor<4x2x3xf16>}> : () -> tensor<4x2x3xf16>
    "func.return"(%0) : (tensor<4x2x3xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

