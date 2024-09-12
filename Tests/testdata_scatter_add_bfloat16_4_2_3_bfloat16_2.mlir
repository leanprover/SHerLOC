"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[3, 2]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3xbf16>, tensor<2xbf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3xbf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%7) : (tensor<bf16>) -> ()
    }) : (tensor<4x2x3xbf16>, tensor<2xi64>, tensor<2xbf16>) -> tensor<4x2x3xbf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x2x3xbf16>, tensor<4x2x3xbf16>) -> ()
    "func.return"(%6) : (tensor<4x2x3xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3xbf16>, tensor<2xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[1.609380e+00, -8.812500e+00, 3.769530e-01], [1.336670e-02, 3.312500e+00, -3.265630e+00]], [[-8.000000e+00, 1.359380e+00, 1.171880e+00], [-2.156250e+00, -8.625000e+00, 2.593750e+00]], [[-4.281250e+00, 4.187500e+00, 1.859380e+00], [-6.484380e-01, -8.687500e+00, -3.250000e+00]], [[-2.859380e+00, 3.203130e+00, -1.593750e+00], [9.960930e-01, 2.687500e+00, 5.781250e-01]]]> : tensor<4x2x3xbf16>}> : () -> tensor<4x2x3xbf16>
    %2 = "stablehlo.constant"() <{value = dense<[-3.093750e+00, -2.753910e-01]> : tensor<2xbf16>}> : () -> tensor<2xbf16>
    "func.return"(%1, %2) : (tensor<4x2x3xbf16>, tensor<2xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[1.609380e+00, -8.812500e+00, 3.769530e-01], [1.336670e-02, 3.312500e+00, -3.265630e+00]], [[-8.000000e+00, 1.359380e+00, 1.171880e+00], [-2.156250e+00, -8.625000e+00, 2.593750e+00]], [[-4.281250e+00, 4.187500e+00, 1.859380e+00], [-6.484380e-01, -8.687500e+00, -3.250000e+00]], [[-2.859380e+00, 3.203130e+00, -4.687500e+00], [9.960930e-01, 2.687500e+00, 3.027340e-01]]]> : tensor<4x2x3xbf16>}> : () -> tensor<4x2x3xbf16>
    "func.return"(%0) : (tensor<4x2x3xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

