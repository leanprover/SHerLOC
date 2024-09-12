"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x2x3xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x2x3xf64>, tensor<2x3xf64>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x2x3xf64>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      "stablehlo.return"(%arg1) : (tensor<f64>) -> ()
    }) : (tensor<1x2x3xf64>, tensor<1xi64>, tensor<2x3xf64>) -> tensor<1x2x3xf64>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<1x2x3xf64>, tensor<1x2x3xf64>) -> ()
    "func.return"(%6) : (tensor<1x2x3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x2x3xf64>, tensor<2x3xf64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[3.0482818399425007, 4.1896917669899496, -3.4307461601612372], [5.5204590309628863, -0.3153938135196856, 3.39187256241937]]]> : tensor<1x2x3xf64>}> : () -> tensor<1x2x3xf64>
    %2 = "stablehlo.constant"() <{value = dense<[[-3.5790082021932568, -0.96731532361838823, -1.3087279865335026], [2.9626371509405915, 1.0945614471023875, -4.1147444060484943]]> : tensor<2x3xf64>}> : () -> tensor<2x3xf64>
    "func.return"(%1, %2) : (tensor<1x2x3xf64>, tensor<2x3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x2x3xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[-3.5790082021932568, -0.96731532361838823, -1.3087279865335026], [2.9626371509405915, 1.0945614471023875, -4.1147444060484943]]]> : tensor<1x2x3xf64>}> : () -> tensor<1x2x3xf64>
    "func.return"(%0) : (tensor<1x2x3xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

