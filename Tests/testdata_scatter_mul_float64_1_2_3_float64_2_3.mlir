"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x2x3xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x2x3xf64>, tensor<2x3xf64>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x2x3xf64>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      "stablehlo.return"(%7) : (tensor<f64>) -> ()
    }) : (tensor<1x2x3xf64>, tensor<1xi64>, tensor<2x3xf64>) -> tensor<1x2x3xf64>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<1x2x3xf64>, tensor<1x2x3xf64>) -> ()
    "func.return"(%6) : (tensor<1x2x3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x2x3xf64>, tensor<2x3xf64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[-1.9496638195342539, -5.4588870676814425, 4.1749035380881958], [-4.9247705077788364, -5.1237762282540311, 0.47912346364803982]]]> : tensor<1x2x3xf64>}> : () -> tensor<1x2x3xf64>
    %2 = "stablehlo.constant"() <{value = dense<[[-2.0849539812706688, -1.8483073442031195, 3.9152889538532589], [-3.135714283477232, 2.1974411834856986, -4.9377501078138355]]> : tensor<2x3xf64>}> : () -> tensor<2x3xf64>
    "func.return"(%1, %2) : (tensor<1x2x3xf64>, tensor<2x3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x2x3xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[4.0649593426773212, 10.089701058371041, 16.345953706079602], [15.442673224089518, -11.259196898930426, -2.3657919342842471]]]> : tensor<1x2x3xf64>}> : () -> tensor<1x2x3xf64>
    "func.return"(%0) : (tensor<1x2x3xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

