"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[1], [0], [1]]> : tensor<3x1xi64>}> : () -> tensor<3x1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<3xf16>, tensor<3xf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<3xf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
      "stablehlo.return"(%7) : (tensor<f16>) -> ()
    }) : (tensor<3xf16>, tensor<3x1xi64>, tensor<3xf16>) -> tensor<3xf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3xf16>, tensor<3xf16>) -> ()
    "func.return"(%6) : (tensor<3xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<3xf16>, tensor<3xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[-3.320310e-02, 6.083980e-01, 2.894530e+00]> : tensor<3xf16>}> : () -> tensor<3xf16>
    %2 = "stablehlo.constant"() <{value = dense<[1.019530e+00, 3.232420e+00, -2.771000e-01]> : tensor<3xf16>}> : () -> tensor<3xf16>
    "func.return"(%1, %2) : (tensor<3xf16>, tensor<3xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-3.320310e-02, -2.771000e-01, 2.894530e+00]> : tensor<3xf16>}> : () -> tensor<3xf16>
    "func.return"(%0) : (tensor<3xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

