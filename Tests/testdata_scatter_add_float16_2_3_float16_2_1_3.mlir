"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<2> : tensor<1x3x1xi64>}> : () -> tensor<1x3x1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3xf16>, tensor<2x1x3xf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
      "stablehlo.return"(%7) : (tensor<f16>) -> ()
    }) : (tensor<2x3xf16>, tensor<1x3x1xi64>, tensor<2x1x3xf16>) -> tensor<2x3xf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x3xf16>, tensor<2x3xf16>) -> ()
    "func.return"(%6) : (tensor<2x3xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3xf16>, tensor<2x1x3xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-2.673830e+00, 3.757810e+00, 4.321290e-01], [6.704100e-01, 1.030270e-01, -2.166020e+00]]> : tensor<2x3xf16>}> : () -> tensor<2x3xf16>
    %2 = "stablehlo.constant"() <{value = dense<[[[3.265630e+00, 2.517580e+00, 4.003910e+00]], [[-5.527340e-01, -7.543940e-01, -1.539060e+00]]]> : tensor<2x1x3xf16>}> : () -> tensor<2x1x3xf16>
    "func.return"(%1, %2) : (tensor<2x3xf16>, tensor<2x1x3xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-2.673830e+00, 3.757810e+00, 1.021880e+01], [6.704100e-01, 1.030270e-01, -5.011720e+00]]> : tensor<2x3xf16>}> : () -> tensor<2x3xf16>
    "func.return"(%0) : (tensor<2x3xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

