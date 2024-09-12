"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x2x3xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[1, 2]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x2x3xf64>, tensor<1xf64>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x2x3xf64>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      "stablehlo.return"(%arg1) : (tensor<f64>) -> ()
    }) : (tensor<1x2x3xf64>, tensor<2xi64>, tensor<1xf64>) -> tensor<1x2x3xf64>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<1x2x3xf64>, tensor<1x2x3xf64>) -> ()
    "func.return"(%6) : (tensor<1x2x3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x2x3xf64>, tensor<1xf64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[3.1180636631305356, -0.59281282974617866, 5.4959926326093509], [-1.8954874077072306, 1.099568710576964, 2.1377549319651732]]]> : tensor<1x2x3xf64>}> : () -> tensor<1x2x3xf64>
    %2 = "stablehlo.constant"() <{value = dense<-2.6068046827223403> : tensor<1xf64>}> : () -> tensor<1xf64>
    "func.return"(%1, %2) : (tensor<1x2x3xf64>, tensor<1xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x2x3xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[3.1180636631305356, -0.59281282974617866, 5.4959926326093509], [-1.8954874077072306, 1.099568710576964, -2.6068046827223403]]]> : tensor<1x2x3xf64>}> : () -> tensor<1x2x3xf64>
    "func.return"(%0) : (tensor<1x2x3xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

