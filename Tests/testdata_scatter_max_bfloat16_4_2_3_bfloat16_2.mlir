"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[3, 2]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3xbf16>, tensor<2xbf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3xbf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%7) : (tensor<bf16>) -> ()
    }) : (tensor<4x2x3xbf16>, tensor<2xi64>, tensor<2xbf16>) -> tensor<4x2x3xbf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x2x3xbf16>, tensor<4x2x3xbf16>) -> ()
    "func.return"(%6) : (tensor<4x2x3xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3xbf16>, tensor<2xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[-9.648430e-01, 1.664060e+00, -9.062500e-01], [5.507810e-01, -2.390630e+00, -7.265630e-01]], [[1.429690e+00, 1.007810e+00, 1.648440e+00], [1.742190e+00, -2.187500e+00, -2.406250e+00]], [[-3.187500e+00, 7.773430e-01, -9.218750e-01], [-2.609380e+00, -1.390630e+00, 3.937500e+00]], [[2.546880e+00, 3.535160e-01, -5.546880e-01], [4.312500e+00, -3.203130e+00, 2.984380e+00]]]> : tensor<4x2x3xbf16>}> : () -> tensor<4x2x3xbf16>
    %2 = "stablehlo.constant"() <{value = dense<[-2.093750e+00, -2.765630e+00]> : tensor<2xbf16>}> : () -> tensor<2xbf16>
    "func.return"(%1, %2) : (tensor<4x2x3xbf16>, tensor<2xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[-9.648430e-01, 1.664060e+00, -9.062500e-01], [5.507810e-01, -2.390630e+00, -7.265630e-01]], [[1.429690e+00, 1.007810e+00, 1.648440e+00], [1.742190e+00, -2.187500e+00, -2.406250e+00]], [[-3.187500e+00, 7.773430e-01, -9.218750e-01], [-2.609380e+00, -1.390630e+00, 3.937500e+00]], [[2.546880e+00, 3.535160e-01, -5.546880e-01], [4.312500e+00, -3.203130e+00, 2.984380e+00]]]> : tensor<4x2x3xbf16>}> : () -> tensor<4x2x3xbf16>
    "func.return"(%0) : (tensor<4x2x3xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

