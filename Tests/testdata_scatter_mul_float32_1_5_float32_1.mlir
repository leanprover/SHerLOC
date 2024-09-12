"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x5xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<10> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x5xf32>, tensor<1xf32>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x5xf32>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%7) : (tensor<f32>) -> ()
    }) : (tensor<1x5xf32>, tensor<1xi64>, tensor<1xf32>) -> tensor<1x5xf32>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<1x5xf32>, tensor<1x5xf32>) -> ()
    "func.return"(%6) : (tensor<1x5xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x5xf32>, tensor<1xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[3.88604307, 1.6223681, -1.57475483, -0.727199554, -3.12737036]]> : tensor<1x5xf32>}> : () -> tensor<1x5xf32>
    %2 = "stablehlo.constant"() <{value = dense<-4.81208467> : tensor<1xf32>}> : () -> tensor<1xf32>
    "func.return"(%1, %2) : (tensor<1x5xf32>, tensor<1xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x5xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[3.88604307, 1.6223681, -1.57475483, -0.727199554, -3.12737036]]> : tensor<1x5xf32>}> : () -> tensor<1x5xf32>
    "func.return"(%0) : (tensor<1x5xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

