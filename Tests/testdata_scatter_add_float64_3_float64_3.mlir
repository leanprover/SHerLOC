"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[1], [0], [1]]> : tensor<3x1xi64>}> : () -> tensor<3x1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<3xf64>, tensor<3xf64>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<3xf64>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      "stablehlo.return"(%7) : (tensor<f64>) -> ()
    }) : (tensor<3xf64>, tensor<3x1xi64>, tensor<3xf64>) -> tensor<3xf64>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3xf64>, tensor<3xf64>) -> ()
    "func.return"(%6) : (tensor<3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<3xf64>, tensor<3xf64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[-5.0069781707696936, -1.7734574356835437, -0.7866993300367906]> : tensor<3xf64>}> : () -> tensor<3xf64>
    %2 = "stablehlo.constant"() <{value = dense<[3.323503884427585, 3.5010829823036955, -1.5196055749398507]> : tensor<3xf64>}> : () -> tensor<3xf64>
    "func.return"(%1, %2) : (tensor<3xf64>, tensor<3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-1.5058951884659981, 0.030440873804190582, -0.7866993300367906]> : tensor<3xf64>}> : () -> tensor<3xf64>
    "func.return"(%0) : (tensor<3xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

