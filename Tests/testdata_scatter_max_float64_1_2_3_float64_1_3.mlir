"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x2x3xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<1> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x2x3xf64>, tensor<1x3xf64>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x2x3xf64>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      "stablehlo.return"(%7) : (tensor<f64>) -> ()
    }) : (tensor<1x2x3xf64>, tensor<1xi64>, tensor<1x3xf64>) -> tensor<1x2x3xf64>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<1x2x3xf64>, tensor<1x2x3xf64>) -> ()
    "func.return"(%6) : (tensor<1x2x3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x2x3xf64>, tensor<1x3xf64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[1.8071611277651978, 4.4744017101386575, -4.6638976477688674], [2.7499872724688186, 1.6802883397705222, 3.5587056100052568]]]> : tensor<1x2x3xf64>}> : () -> tensor<1x2x3xf64>
    %2 = "stablehlo.constant"() <{value = dense<[[-1.7860850879846581, -1.3426974404518877, 3.7600261949158686]]> : tensor<1x3xf64>}> : () -> tensor<1x3xf64>
    "func.return"(%1, %2) : (tensor<1x2x3xf64>, tensor<1x3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x2x3xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[1.8071611277651978, 4.4744017101386575, -4.6638976477688674], [2.7499872724688186, 1.6802883397705222, 3.7600261949158686]]]> : tensor<1x2x3xf64>}> : () -> tensor<1x2x3xf64>
    "func.return"(%0) : (tensor<1x2x3xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

