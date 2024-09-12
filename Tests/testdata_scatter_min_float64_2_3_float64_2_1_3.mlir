"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<2> : tensor<1x3x1xi64>}> : () -> tensor<1x3x1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3xf64>, tensor<2x1x3xf64>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xf64>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      "stablehlo.return"(%7) : (tensor<f64>) -> ()
    }) : (tensor<2x3xf64>, tensor<1x3x1xi64>, tensor<2x1x3xf64>) -> tensor<2x3xf64>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x3xf64>, tensor<2x3xf64>) -> ()
    "func.return"(%6) : (tensor<2x3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3xf64>, tensor<2x1x3xf64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0.36122310516137945, 2.6986009434872651, 5.4751384889125818], [-0.23399422136522924, -1.2078836128544281, -2.2954949302425307]]> : tensor<2x3xf64>}> : () -> tensor<2x3xf64>
    %2 = "stablehlo.constant"() <{value = dense<[[[-1.4222101237409046, 2.2291616554877445, 1.0076255190643058]], [[5.8557965706290176, -2.4118192541735386, -2.225702241190167]]]> : tensor<2x1x3xf64>}> : () -> tensor<2x1x3xf64>
    "func.return"(%1, %2) : (tensor<2x3xf64>, tensor<2x1x3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[0.36122310516137945, 2.6986009434872651, -1.4222101237409046], [-0.23399422136522924, -1.2078836128544281, -2.4118192541735386]]> : tensor<2x3xf64>}> : () -> tensor<2x3xf64>
    "func.return"(%0) : (tensor<2x3xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

